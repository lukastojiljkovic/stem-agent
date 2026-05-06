"""Tests for the type system used by the pipeline validator."""
import pytest
from stem_agent.types import (
    TypeName,
    can_coerce,
    is_compatible,
)


def test_type_name_basic_members():
    assert TypeName.QUERY in TypeName
    assert TypeName.DOCUMENTS in TypeName
    assert TypeName.CLAUSES in TypeName
    assert TypeName.PDF_PATH in TypeName


def test_strict_compatibility_identical_types():
    assert is_compatible(TypeName.TEXT, TypeName.TEXT)


def test_strict_compatibility_different_types():
    assert not is_compatible(TypeName.QUERY, TypeName.CLAUSES)


def test_documents_to_text_coercion():
    assert can_coerce(TypeName.DOCUMENTS, TypeName.TEXT)


def test_document_singular_to_text_coercion():
    assert can_coerce(TypeName.DOCUMENT, TypeName.TEXT)


def test_no_silent_text_to_documents():
    assert not can_coerce(TypeName.TEXT, TypeName.DOCUMENTS)


def test_is_compatible_uses_coercion():
    assert is_compatible(TypeName.DOCUMENT, TypeName.TEXT)
    assert not is_compatible(TypeName.SCORE, TypeName.PDF_PATH)
