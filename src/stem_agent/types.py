"""Discriminated type system for typed pipeline validation.

Producers and consumers of pipeline steps are matched on output_type → input_type.
A small set of safe coercions is allowed (Document/Documents → Text, container → element).
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class TypeName(str, Enum):
    QUERY = "Query"
    DOCUMENTS = "Documents"
    DOCUMENT = "Document"
    TEXT = "Text"
    TIME_SERIES = "TimeSeries"
    FILING = "Filing"
    STRUCTURED_DATA = "StructuredData"
    INDICATORS = "Indicators"
    TRENDS = "Trends"
    CORR_MATRIX = "CorrMatrix"
    RATIO_REPORT = "RatioReport"
    CLAUSES = "Clauses"
    OBLIGATION_LIST = "ObligationList"
    RULE_HITS = "RuleHits"
    ENTITIES = "Entities"
    LABEL = "Label"
    COMPARISON = "Comparison"
    ISSUES = "Issues"
    REASONING_TRACE = "ReasoningTrace"
    EXEC_RESULT = "ExecResult"
    SCORE = "Score"
    TEX_PROJECT = "TexProject"
    PDF_PATH = "PdfPath"
    ISSUES_LIST = "IssuesList"
    CASE_DOCS = "CaseDocs"
    LEGAL_TEXT = "LegalText"


_COERCIONS: dict[TypeName, set[TypeName]] = {
    # Container/document types
    TypeName.DOCUMENTS: {TypeName.TEXT, TypeName.DOCUMENT},
    TypeName.DOCUMENT: {TypeName.TEXT},
    TypeName.CASE_DOCS: {TypeName.DOCUMENTS, TypeName.TEXT},
    TypeName.LEGAL_TEXT: {TypeName.TEXT},
    TypeName.FILING: {TypeName.STRUCTURED_DATA, TypeName.TEXT},
    # Structured outputs that are dict-like and stringifiable
    TypeName.CLAUSES: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.OBLIGATION_LIST: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.RULE_HITS: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.INDICATORS: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.TRENDS: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.CORR_MATRIX: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.RATIO_REPORT: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.ENTITIES: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.COMPARISON: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.REASONING_TRACE: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.EXEC_RESULT: {TypeName.TEXT, TypeName.STRUCTURED_DATA},
    TypeName.ISSUES: {TypeName.TEXT},
    TypeName.LABEL: {TypeName.TEXT},
    TypeName.SCORE: {TypeName.TEXT},
}


def can_coerce(src: TypeName, dst: TypeName) -> bool:
    return dst in _COERCIONS.get(src, set())


def is_compatible(producer: TypeName, consumer: TypeName) -> bool:
    """A producer can feed a consumer iff types match exactly or via allowed coercion."""
    return producer is consumer or can_coerce(producer, consumer)


@dataclass(frozen=True)
class TypedValue:
    """A runtime value paired with its declared TypeName, for checked passing."""
    type_name: TypeName
    value: Any
    meta: dict[str, Any] = field(default_factory=dict)
