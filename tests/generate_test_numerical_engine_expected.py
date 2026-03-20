"""Generate expected comparison results for ``tests/test_numerical_engine.py``."""

from __future__ import annotations

import json

from tests.official_numerical_engine_suite import (
    EXPECTED_RESULTS_PATH,
    build_expected_results_document,
)


def main() -> None:
    EXPECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = build_expected_results_document()
    EXPECTED_RESULTS_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
