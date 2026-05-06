"""Single entrypoint that registers every primitive (and universal) into a ToolLibrary instance."""
from __future__ import annotations

from . import retrieval as R
from . import processing as P
from . import reasoning as RS
from . import domain_econ as DE
from . import domain_legal as DL
from . import execution as EX
from . import evaluation as EV
from .registry import ToolLibrary


_PRIMITIVE_FNS = [
    R.web_search, R.wikipedia_lookup, R.wikipedia_search, R.arxiv_search,
    R.semantic_scholar_search, R.openalex_search, R.extract_search_query,
    R.fred_query, R.edgar_fetch, R.courtlistener_search, R.eurlex_lookup,
    P.pdf_extract, P.html_extract, P.summarize, P.extract_entities, P.classify, P.normalize_data,
    RS.chain_of_thought, RS.compare, RS.detect_inconsistencies,
    DE.compute_indicators, DE.trend_analysis, DE.correlation_analysis, DE.financial_ratios,
    DL.clause_extraction, DL.obligation_detection, DL.rule_matching,
    EX.python_exec,
    EV.score_accuracy, EV.consistency_check, EV.completeness_check,
]


def register_all_primitives(library: ToolLibrary) -> int:
    n = 0
    for fn in _PRIMITIVE_FNS:
        library.register(fn.tool)
        n += 1
    return n


def register_all_universal(library: ToolLibrary) -> int:
    """Register universal frozen tools. Imported lazily so that early tests
    that don't need them still work."""
    from .universal import latex_builder as ULB
    from .universal import grammar as UG
    from .universal import pdf as UP
    from .universal import report_finalize as URF
    fns = [
        ULB.latex_init, ULB.latex_section, ULB.latex_table, ULB.latex_chart,
        UG.grammar_check, UP.pdf_compile, URF.report_finalize,
    ]
    n = 0
    for fn in fns:
        library.register(fn.tool)
        n += 1
    return n


def register_all(library: ToolLibrary) -> tuple[int, int]:
    p = register_all_primitives(library)
    u = register_all_universal(library)
    return p, u
