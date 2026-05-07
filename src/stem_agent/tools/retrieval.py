"""Retrieval primitives. All return plain dicts/lists; the executor wraps them.

Tool list:
    web_search           — Tavily (if TAVILY_API_KEY) -> DDG fallback -> empty.
    wikipedia_lookup     — wikipedia-api.
    wikipedia_search     — full-text search the Wikipedia API (no key, returns multiple hits).
    arxiv_search         — arxiv pkg with built-in 1-req/3-sec rate limit.
    semantic_scholar_search — Semantic Scholar Graph API (no key, public rate limits).
    openalex_search      — OpenAlex /works endpoint (no key, polite-pool rate limits).
    extract_search_query — LLM-driven Text -> Query bridge: lets evolution propose
                           web/wiki/arxiv/SS/OpenAlex steps after a TEXT input.
    fred_query           — fredapi (requires FRED_API_KEY).
    edgar_fetch          — edgartools (User-Agent set; 10 req/sec respected).
    courtlistener_search — raw requests + COURTLISTENER_TOKEN.
    eurlex_lookup        — raw requests + BeautifulSoup over EUR-Lex.

The first six (web/wikipedia*/arxiv/semantic_scholar/openalex) and
extract_search_query are KEY-FREE and form the agent's no-credentials
research path: TEXT task content -> extract_search_query -> {wiki, arxiv,
SS, OpenAlex, web} -> Documents -> summarize.
"""
from __future__ import annotations

import html
import os
from typing import Any

import requests

from ..config import has_tavily_key, has_fred_key, has_courtlistener_key
from ..llm.lm_client import LMClient, ChatMessage, clean_llm_query
from ..types import TypeName
from ..ui.console import log_retrieve, log_warn
from ._cache import JsonCache
from .base import tool, ToolKind

# Module-level lazy singleton for the LM client (mirrors processing.py / reasoning.py).
_LM: LMClient | None = None


def _lm() -> LMClient:
    global _LM
    if _LM is None:
        _LM = LMClient()
    return _LM


# Shared HTTP session: reuses TCP+TLS connections across calls. Gives every
# key-free retrieval primitive a polite User-Agent without each tool repeating it.
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "stem-agent/0.1 (research; mailto:noreply@example.com)"})

WEB_CACHE = JsonCache("web", ttl_s=24 * 3600)
WIKI_CACHE = JsonCache("wiki", ttl_s=7 * 24 * 3600)
WIKI_SEARCH_CACHE = JsonCache("wiki_search", ttl_s=24 * 3600)
ARXIV_CACHE = JsonCache("arxiv", ttl_s=7 * 24 * 3600)
SS_CACHE = JsonCache("semantic_scholar", ttl_s=7 * 24 * 3600)
OPENALEX_CACHE = JsonCache("openalex", ttl_s=7 * 24 * 3600)
QUERY_REWRITE_CACHE = JsonCache("query_rewrite", ttl_s=24 * 3600)
FRED_CACHE = JsonCache("fred", ttl_s=24 * 3600)
EDGAR_CACHE = JsonCache("edgar", ttl_s=7 * 24 * 3600)
CL_CACHE = JsonCache("courtlistener", ttl_s=7 * 24 * 3600)
EURLEX_CACHE = JsonCache("eurlex", ttl_s=7 * 24 * 3600)


@tool(
    name="web_search",
    description="General web search. Returns a list of {title,url,snippet}. Tavily preferred, DDG fallback.",
    input_type=TypeName.QUERY,
    output_type=TypeName.DOCUMENTS,
    kind=ToolKind.PRIMITIVE,
    cost=0.10,
)
def web_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    cached = WEB_CACHE.get({"q": query, "k": k})
    if cached is not None:
        log_retrieve(f"web_search (cache hit) k={k}: {query!r}")
        return cached
    log_retrieve(f"web_search k={k}: {query!r}")
    docs: list[dict[str, Any]] = []
    if has_tavily_key():
        try:
            from tavily import TavilyClient
            tv = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            res = tv.search(query=query, max_results=k, search_depth="basic")
            for r in res.get("results", []):
                docs.append({"title": r.get("title",""), "url": r.get("url",""),
                             "snippet": (r.get("content") or "")[:1200]})
        except Exception as e:
            log_warn(f"Tavily failed: {e}; falling back to DDG")
    if not docs:
        try:
            from ddgs import DDGS
            with DDGS() as d:
                for r in d.text(query, max_results=k):
                    docs.append({"title": r.get("title",""), "url": r.get("href") or r.get("url",""),
                                 "snippet": (r.get("body") or "")[:1200]})
        except Exception as e:
            log_warn(f"DDG failed: {e}")
    WEB_CACHE.put({"q": query, "k": k}, docs)
    return docs


@tool(
    name="wikipedia_lookup",
    description="Fetch the lead section of a Wikipedia article. Returns {title,url,text}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.DOCUMENT,
    kind=ToolKind.PRIMITIVE,
    cost=0.05,
)
def wikipedia_lookup(title: str, lang: str = "en", max_chars: int = 4000) -> dict[str, Any]:
    cached = WIKI_CACHE.get({"t": title, "lang": lang, "n": max_chars})
    if cached is not None:
        log_retrieve(f"wikipedia (cache) {title!r} lang={lang}")
        return cached
    log_retrieve(f"wikipedia {title!r} lang={lang}")
    out: dict[str, Any] = {"title": title, "url": "", "text": ""}
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(language=lang, user_agent="stem-agent/0.1 (research)")
        page = wiki.page(title)
        if page.exists():
            out = {"title": page.title, "url": page.fullurl, "text": (page.summary or page.text)[:max_chars]}
    except Exception as e:
        log_warn(f"Wikipedia failed: {e}")
    WIKI_CACHE.put({"t": title, "lang": lang, "n": max_chars}, out)
    return out


@tool(
    name="arxiv_search",
    description="Search arXiv. Returns list of {title,authors,abstract,url,published}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.DOCUMENTS,
    kind=ToolKind.PRIMITIVE,
    cost=0.10,
)
def arxiv_search(query: str, k: int = 3) -> list[dict[str, Any]]:
    cached = ARXIV_CACHE.get({"q": query, "k": k})
    if cached is not None:
        log_retrieve(f"arxiv (cache) k={k}: {query!r}")
        return cached
    log_retrieve(f"arxiv k={k}: {query!r}")
    docs: list[dict[str, Any]] = []
    try:
        import arxiv
        search = arxiv.Search(query=query, max_results=k, sort_by=arxiv.SortCriterion.Relevance)
        client = arxiv.Client(page_size=k, delay_seconds=3.0, num_retries=1)
        for r in client.results(search):
            docs.append({
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "abstract": (r.summary or "")[:1500],
                "url": r.entry_id,
                "published": str(r.published.date()) if r.published else "",
            })
    except Exception as e:
        log_warn(f"arxiv failed: {e}")
    ARXIV_CACHE.put({"q": query, "k": k}, docs)
    return docs


@tool(
    name="fred_query",
    description="Fetch a FRED time series. Returns {series_id, dates, values, freq}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.TIME_SERIES,
    kind=ToolKind.PRIMITIVE,
    domain="economics",
    cost=0.05,
)
def fred_query(series_id: str, observation_start: str | None = None,
               observation_end: str | None = None) -> dict[str, Any]:
    if not has_fred_key():
        return {"error": "FRED_API_KEY not set", "series_id": series_id, "dates": [], "values": []}
    payload = {"sid": series_id, "s": observation_start or "", "e": observation_end or ""}
    cached = FRED_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"FRED (cache) {series_id} {observation_start}..{observation_end}")
        return cached
    log_retrieve(f"FRED {series_id} {observation_start}..{observation_end}")
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.environ["FRED_API_KEY"])
        s = fred.get_series(series_id, observation_start=observation_start,
                            observation_end=observation_end)
        out = {
            "series_id": series_id,
            "dates": [str(d.date()) for d in s.index.to_pydatetime()],
            "values": [float(v) if v == v else None for v in s.values],
            "freq": "unknown",
        }
    except Exception as e:
        log_warn(f"FRED failed: {e}")
        out = {"error": str(e), "series_id": series_id, "dates": [], "values": []}
    FRED_CACHE.put(payload, out)
    return out


_XBRL_CONCEPT_ALIASES: dict[str, list[str]] = {
    # Different filers report the same fact under slightly different XBRL concept
    # names; we try the most-specific names first then fall back to broader ones.
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_liabilities": ["Liabilities", "LiabilitiesAndStockholdersEquity"],
    "stockholders_equity": ["StockholdersEquity",
                            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "operating_income": ["OperatingIncomeLoss"],
    "cash_from_ops": ["NetCashProvidedByUsedInOperatingActivities"],
}


def _try_concept_chain(financials, names: list[str]) -> float | None:
    for name in names:
        v = _try_value(financials, name)
        if v is not None:
            return v
    return None


@tool(
    name="edgar_fetch",
    description="Fetch a SEC filing by ticker+form+year. Returns {ticker, form, accession, text, xbrl_facts}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.FILING,
    kind=ToolKind.PRIMITIVE,
    domain="economics",
    cost=0.20,
)
def edgar_fetch(ticker: str, form: str = "10-K", year: int | None = None) -> dict[str, Any]:
    payload = {"t": ticker, "f": form, "y": year or 0}
    cached = EDGAR_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"EDGAR (cache) {ticker} {form} {year}")
        return cached
    log_retrieve(f"EDGAR {ticker} {form} {year}")
    out: dict[str, Any] = {"ticker": ticker, "form": form, "accession": "", "text": "", "xbrl_facts": {}}
    try:
        from edgar import set_identity, Company
        # SEC requires a contact User-Agent. The default email is example.com,
        # which SEC accepts (RFC 2606 reserved); users with heavy use should set
        # EDGAR_USER_AGENT to their real contact info.
        set_identity(os.environ.get("EDGAR_USER_AGENT", "Stem Agent contact@example.com"))
        c = Company(ticker)
        filings = c.get_filings(form=form)
        if year:
            try:
                filings = filings.filter(date=f"{year}-01-01:{year}-12-31")
            except Exception:
                pass
        if not filings:
            return out
        f = filings.latest()
        out["accession"] = getattr(f, "accession_number", "") or ""
        try:
            out["text"] = (f.text() or "")[:20000]
        except Exception:
            out["text"] = ""
        try:
            obj = f.obj()
            fin = getattr(obj, "financials", None) if obj is not None else None
            if fin is not None:
                facts: dict[str, float | None] = {}
                for label, names in _XBRL_CONCEPT_ALIASES.items():
                    facts[label] = _try_concept_chain(fin, names)
                out["xbrl_facts"] = facts
        except Exception as e:
            log_warn(f"EDGAR XBRL extraction failed: {e}")
    except Exception as e:
        log_warn(f"EDGAR fetch failed: {e}")
        out["error"] = str(e)
    EDGAR_CACHE.put(payload, out)
    return out


def _try_value(financials, concept: str) -> float | None:
    try:
        v = financials.get(concept)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


@tool(
    name="courtlistener_search",
    description="Free-text search over US case-law via CourtListener. Returns {results:[{caseName,absolute_url,plain_text}]}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.CASE_DOCS,
    kind=ToolKind.PRIMITIVE,
    domain="legal",
    cost=0.20,
)
def courtlistener_search(query: str, k: int = 5) -> dict[str, Any]:
    payload = {"q": query, "k": k}
    cached = CL_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"CourtListener (cache) {query!r}")
        return cached
    log_retrieve(f"CourtListener {query!r}")
    out: dict[str, Any] = {"results": []}
    if not has_courtlistener_key():
        out["error"] = "COURTLISTENER_TOKEN not set"
        return out
    try:
        headers = {"Authorization": f"Token {os.environ['COURTLISTENER_TOKEN']}"}
        url = "https://www.courtlistener.com/api/rest/v4/search/"
        r = _SESSION.get(url, params={"q": query, "type": "o", "page_size": k}, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json().get("results", [])[:k]
            for item in data:
                out["results"].append({
                    "caseName": item.get("caseName") or item.get("caseNameShort", ""),
                    "absolute_url": "https://www.courtlistener.com" + (item.get("absolute_url") or ""),
                    "plain_text": (item.get("plain_text") or "")[:6000],
                    "court": item.get("court", ""),
                    "dateFiled": item.get("dateFiled", ""),
                })
        else:
            out["error"] = f"HTTP {r.status_code}"
    except Exception as e:
        log_warn(f"CourtListener failed: {e}")
        out["error"] = str(e)
    CL_CACHE.put(payload, out)
    return out


@tool(
    name="eurlex_lookup",
    description="Look up a CELEX id (or free text) on EUR-Lex via CELLAR SPARQL. Returns {celex,title,full_text}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.LEGAL_TEXT,
    kind=ToolKind.PRIMITIVE,
    domain="legal",
    cost=0.20,
)
def eurlex_lookup(celex_or_query: str) -> dict[str, Any]:
    payload = {"k": celex_or_query}
    cached = EURLEX_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"EUR-Lex (cache) {celex_or_query!r}")
        return cached
    log_retrieve(f"EUR-Lex {celex_or_query!r}")
    out: dict[str, Any] = {"celex": "", "title": "", "full_text": ""}
    try:
        if celex_or_query.upper().startswith(("3", "5", "6", "1", "2")) and len(celex_or_query) >= 8:
            celex = celex_or_query
            url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
            r = _SESSION.get(url, timeout=20)
            if r.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "lxml")
                title = soup.find("title")
                out["celex"] = celex
                out["title"] = (title.text if title else "")[:300]
                main = soup.select_one("#TexteOnly") or soup.select_one("#text") or soup
                out["full_text"] = main.get_text(" ", strip=True)[:15000]
        else:
            r = _SESSION.get(
                "https://eur-lex.europa.eu/search.html",
                params={"text": celex_or_query, "qid": ""},
                timeout=20,
            )
            out["title"] = "(search results page; agent should follow links)"
            out["full_text"] = (r.text or "")[:6000]
    except Exception as e:
        log_warn(f"EUR-Lex failed: {e}")
        out["error"] = str(e)
    EURLEX_CACHE.put(payload, out)
    return out


# ---------------------------------------------------------------------------
# Key-free research path: extra retrieval primitives that need no credentials.
# ---------------------------------------------------------------------------

@tool(
    name="wikipedia_search",
    description="Full-text search over Wikipedia (no key). Returns list of {title,url,snippet} hits.",
    input_type=TypeName.QUERY,
    output_type=TypeName.DOCUMENTS,
    kind=ToolKind.PRIMITIVE,
    cost=0.05,
)
def wikipedia_search(query: str, k: int = 5, lang: str = "en") -> list[dict[str, Any]]:
    payload = {"q": query, "k": k, "lang": lang}
    cached = WIKI_SEARCH_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"wiki_search (cache) k={k}: {query!r}")
        return cached
    log_retrieve(f"wiki_search k={k}: {query!r}")
    out: list[dict[str, Any]] = []
    try:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": str(k), "format": "json", "utf8": "1",
        }
        r = _SESSION.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json().get("query", {}).get("search", [])
            for item in data[:k]:
                title = item.get("title", "")
                raw_snip = (item.get("snippet") or "").replace('<span class="searchmatch">', "").replace("</span>", "")
                # Wikipedia search snippets contain HTML entities (&quot;, &amp;, ...)
                # and occasional inline tags; decode + strip residue.
                snippet = html.unescape(raw_snip)
                out.append({
                    "title": title,
                    "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "snippet": snippet[:1500],
                })
    except Exception as e:
        log_warn(f"wikipedia_search failed: {e}")
    WIKI_SEARCH_CACHE.put(payload, out)
    return out


@tool(
    name="semantic_scholar_search",
    description="Search Semantic Scholar Graph API (no key). Returns list of {title,abstract,authors,year,url}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.DOCUMENTS,
    kind=ToolKind.PRIMITIVE,
    cost=0.10,
)
def semantic_scholar_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    payload = {"q": query, "k": k}
    cached = SS_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"semantic_scholar (cache) k={k}: {query!r}")
        return cached
    log_retrieve(f"semantic_scholar k={k}: {query!r}")
    out: list[dict[str, Any]] = []
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query, "limit": str(k),
            "fields": "title,abstract,year,authors,url,venue,externalIds",
        }
        r = _SESSION.get(url, params=params, timeout=20)
        if r.status_code == 200:
            for item in (r.json().get("data") or [])[:k]:
                out.append({
                    "title": item.get("title", ""),
                    "abstract": (item.get("abstract") or "")[:1500],
                    "authors": [a.get("name", "") for a in (item.get("authors") or [])][:6],
                    "year": item.get("year"),
                    "venue": item.get("venue", ""),
                    "url": item.get("url") or _ext_url(item.get("externalIds") or {}),
                })
        else:
            log_warn(f"semantic_scholar HTTP {r.status_code}")
    except Exception as e:
        log_warn(f"semantic_scholar failed: {e}")
    SS_CACHE.put(payload, out)
    return out


def _ext_url(ext: dict[str, Any]) -> str:
    if "DOI" in ext: return f"https://doi.org/{ext['DOI']}"
    if "ArXiv" in ext: return f"https://arxiv.org/abs/{ext['ArXiv']}"
    if "PubMed" in ext: return f"https://pubmed.ncbi.nlm.nih.gov/{ext['PubMed']}"
    return ""


@tool(
    name="openalex_search",
    description="Search OpenAlex /works endpoint (no key). Returns list of {title,abstract,authors,year,doi,venue}.",
    input_type=TypeName.QUERY,
    output_type=TypeName.DOCUMENTS,
    kind=ToolKind.PRIMITIVE,
    cost=0.10,
)
def openalex_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    payload = {"q": query, "k": k}
    cached = OPENALEX_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"openalex (cache) k={k}: {query!r}")
        return cached
    log_retrieve(f"openalex k={k}: {query!r}")
    out: list[dict[str, Any]] = []
    try:
        url = "https://api.openalex.org/works"
        params = {"search": query, "per-page": str(k)}
        r = _SESSION.get(url, params=params, timeout=20)
        if r.status_code == 200:
            for w in (r.json().get("results") or [])[:k]:
                # OpenAlex returns abstract as an inverted index; reconstruct it.
                inv = w.get("abstract_inverted_index") or {}
                abstract = _reconstruct_abstract(inv)
                out.append({
                    "title": w.get("title", "") or w.get("display_name", ""),
                    "abstract": (abstract or "")[:1500],
                    "authors": [a.get("author", {}).get("display_name", "")
                                for a in (w.get("authorships") or [])][:6],
                    "year": w.get("publication_year"),
                    "venue": (w.get("host_venue") or {}).get("display_name", ""),
                    "doi": w.get("doi", ""),
                    "url": w.get("doi", "") or w.get("id", ""),
                })
        else:
            log_warn(f"openalex HTTP {r.status_code}")
    except Exception as e:
        log_warn(f"openalex failed: {e}")
    OPENALEX_CACHE.put(payload, out)
    return out


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    if not inverted_index: return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


# ---------------------------------------------------------------------------
# extract_search_query: TEXT -> QUERY bridge. With this primitive, evolution
# can propose pipelines like (pdf_extract -> extract_search_query -> wikipedia_search
# -> summarize) where retrieval happens MID-pipeline rather than only at the
# very start. Without it, TEXT-input tasks have no path to web/wiki/SS/OpenAlex.
# ---------------------------------------------------------------------------

@tool(
    name="extract_search_query",
    description="Distill an input TEXT into a short search-engine-ready Query string. LLM-driven.",
    input_type=TypeName.TEXT,
    output_type=TypeName.QUERY,
    kind=ToolKind.PRIMITIVE,
    cost=0.10,
)
def extract_search_query(text: str, intent: str | None = None,
                         max_words: int = 12) -> str:
    """Produce a focused, keyword-style query that captures the essence of `text`.
    Optional `intent` hint biases toward a domain (e.g. "legal precedents on",
    "financial ratios for")."""
    if not isinstance(max_words, int) or max_words <= 0:
        max_words = 12
    payload = {"t": (text or "")[:2000], "i": intent or "", "n": max_words}
    cached = QUERY_REWRITE_CACHE.get(payload)
    if cached is not None:
        log_retrieve(f"extract_query (cache) -> {cached!r}")
        return str(cached)

    if not text or not str(text).strip():
        return ""
    sys = ("You are a search query distiller. Read the TEXT and produce ONE concise "
           f"keyword-style search query of at most {max_words} words. "
           "No quotes, no Boolean operators, no question marks. "
           "Output ONLY the query, nothing else.")
    user = (f"INTENT: {intent}\n\n" if intent else "") + f"TEXT:\n{str(text)[:2000]}"
    try:
        out = _lm().chat(
            [ChatMessage(role="system", content=sys),
             ChatMessage(role="user", content=user)],
            temperature=0.3, top_p=0.9, max_tokens=64,
        )
        q = clean_llm_query(out.text, max_words=max_words)
    except Exception as e:
        log_warn(f"extract_search_query failed: {e}")
        q = " ".join((text or "").split()[:max_words])
    log_retrieve(f"extract_query -> {q!r}")
    QUERY_REWRITE_CACHE.put(payload, q)
    return q
