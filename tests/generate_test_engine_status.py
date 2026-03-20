"""Generate the markdown status page for ``tests/test_engine_status.py``."""

from __future__ import annotations

from tests.official_numerical_engine_suite import (
    NUMPY_EXPECTED_RESULTS_PATH,
    STATUS_PAGE_PATH,
    load_expected_results,
    render_status_page,
)


def main() -> None:
    STATUS_PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    numpy_document = (
        load_expected_results(NUMPY_EXPECTED_RESULTS_PATH)
        if NUMPY_EXPECTED_RESULTS_PATH.exists()
        else None
    )
    STATUS_PAGE_PATH.write_text(
        render_status_page(load_expected_results(), numpy_document),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
