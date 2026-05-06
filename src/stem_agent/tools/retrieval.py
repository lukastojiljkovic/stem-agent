"""Retrieval primitives. All return plain dicts/lists; the executor wraps them.

Tool list:
    web_search       — Tavily (if TAVILY_API_KEY) → DDG fallback → empty results.
    wikipedia_lookup — wikipedia-api.
    arxiv_search     — arxiv pkg with built-in 1-req/3-sec rate limit.
    fred_query       — fredapi (requires FRED_API_KEY).
    edgar_fetch      — edgartools (User-Agent set; 10 req/sec respected).
    courtlistener_search — raw requests + COURTLISTENER_TOKEN.
    eurlex_lookup    — raw requests + BeautifulSoup over EUR-Lex.
"""
from __future__ import annotations

import os
from typing import Any

from ..config import has_tavily_key, has_fred_key, has_courtlistener_key
from ..types import TypeName
from ..ui.console import log_retrieve, log_warn
from ._cache import JsonCache
from .base import tool, ToolKind

WEB_CACHE = JsonCache("web", ttl_s=24 * 3600)
WIKI_CACHE = JsonCache("wiki", ttl_s=7 * 24 * 3600)
ARXIV_CACHE = JsonCache("arxiv", ttl_s=7 * 24 * 3600)
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
        set_identity(os.environ.get("EDGAR_USER_AGENT", "Stem Agent stemagent@example.com"))
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
            if hasattr(obj, "financials") and obj.financials is not None:
                fin = obj.financials
                out["xbrl_facts"] = {
                    "revenue": _try_value(fin, "Revenues"),
                    "net_income": _try_value(fin, "NetIncomeLoss"),
                    "total_assets": _try_value(fin, "Assets"),
                    "current_assets": _try_value(fin, "AssetsCurrent"),
                    "current_liabilities": _try_value(fin, "LiabilitiesCurrent"),
                    "total_liabilities": _try_value(fin, "Liabilities"),
                    "stockholders_equity": _try_value(fin, "StockholdersEquity"),
                    "operating_income": _try_value(fin, "OperatingIncomeLoss"),
                    "cash_from_ops": _try_value(fin, "NetCashProvidedByUsedInOperatingActivities"),
                }
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
        import requests
        headers = {"Authorization": f"Token {os.environ['COURTLISTENER_TOKEN']}"}
        url = "https://www.courtlistener.com/api/rest/v4/search/"
        r = requests.get(url, params={"q": query, "type": "o", "page_size": k}, headers=headers, timeout=20)
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
        import requests
        if celex_or_query.upper().startswith(("3", "5", "6", "1", "2")) and len(celex_or_query) >= 8:
            celex = celex_or_query
            url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
            r = requests.get(url, timeout=20, headers={"User-Agent": "stem-agent/0.1"})
            if r.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "lxml")
                title = soup.find("title")
                out["celex"] = celex
                out["title"] = (title.text if title else "")[:300]
                main = soup.select_one("#TexteOnly") or soup.select_one("#text") or soup
                out["full_text"] = main.get_text(" ", strip=True)[:15000]
        else:
            r = requests.get(
                "https://eur-lex.europa.eu/search.html",
                params={"text": celex_or_query, "qid": ""},
                timeout=20, headers={"User-Agent": "stem-agent/0.1"},
            )
            out["title"] = "(search results page; agent should follow links)"
            out["full_text"] = (r.text or "")[:6000]
    except Exception as e:
        log_warn(f"EUR-Lex failed: {e}")
        out["error"] = str(e)
    EURLEX_CACHE.put(payload, out)
    return out
