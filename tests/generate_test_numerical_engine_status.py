"""Generate the markdown status page for ``tests/test_numerical_engine.py``."""

from __future__ import annotations

from tests.official_numerical_engine_suite import (
    STATUS_PAGE_PATH,
    load_expected_results,
    render_status_page,
)


def main() -> None:
    STATUS_PAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PAGE_PATH.write_text(
        render_status_page(load_expected_results()),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
