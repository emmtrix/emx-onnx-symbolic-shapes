"""Snapshot test for the generated markdown status page."""

from __future__ import annotations

from tests.official_numerical_engine_suite import (
    NUMPY_EXPECTED_RESULTS_PATH,
    STATUS_PAGE_PATH,
    load_expected_results,
    render_status_page,
)


def test_generated_status_page_matches_snapshot() -> None:
    expected = STATUS_PAGE_PATH.read_text(encoding="utf-8")
    numpy_document = (
        load_expected_results(NUMPY_EXPECTED_RESULTS_PATH)
        if NUMPY_EXPECTED_RESULTS_PATH.exists()
        else None
    )
    actual = render_status_page(load_expected_results(), numpy_document)
    assert actual == expected
