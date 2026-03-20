"""Snapshot test for the generated markdown status page."""

from __future__ import annotations

from tests.official_numerical_engine_suite import (
    STATUS_PAGE_PATH,
    load_expected_results,
    render_status_page,
)


def test_generated_status_page_matches_snapshot() -> None:
    expected = STATUS_PAGE_PATH.read_text(encoding="utf-8")
    actual = render_status_page(load_expected_results())
    assert actual == expected
