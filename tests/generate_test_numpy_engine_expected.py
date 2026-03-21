"""Generate expected comparison results for ``tests/test_numpy_engine.py``."""

from __future__ import annotations

import json

import onnx
from tests.official_numerical_engine_suite import (
    OFFICIAL_TEST_KINDS,
    OfficialTestCase,
    collect_official_test_cases,
)
from tests.test_numpy_engine import (
    EXPECTED_RESULTS_PATH,
    compare_official_case_with_onnx,
)


def build_expected_results_document() -> dict[str, object]:
    """Build an extensible JSON document for all official ONNX tests."""
    cases = collect_official_test_cases()
    tests: dict[str, object] = {}
    for case in cases:
        tests[case.case_id] = _build_expected_test_entry(case)

    return {
        "schema_version": 1,
        "suite": {
            "source": "onnx.backend.test",
            "onnx_version": onnx.__version__,
            "case_count": len(cases),
            "kinds": list(OFFICIAL_TEST_KINDS),
        },
        "tests": tests,
    }


def _build_expected_test_entry(case: OfficialTestCase) -> dict[str, object]:
    return {
        "kind": case.kind,
        "name": case.name,
        "model_name": case.model_name,
        "expected": {
            "comparison_result": compare_official_case_with_onnx(case),
        },
    }


def main() -> None:
    EXPECTED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = build_expected_results_document()
    EXPECTED_RESULTS_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
