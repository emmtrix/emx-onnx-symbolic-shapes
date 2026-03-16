"""Tests for the OTSL shape inference engine."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from oscl.engine import infer_shapes as oscl_infer_shapes
from tests.official_engine_suite import (
    EXPECTED_RESULTS_PATH,
    collect_official_test_cases,
    compare_case,
    get_output_shapes,
    get_output_types,
    load_expected_results,
)

# Suppress numpy overflow warnings emitted during ONNX test-case collection
# (e.g. cast.py, castlike.py). Scoped to numpy's RuntimeWarnings only.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\.")

_OFFICIAL_CASES = collect_official_test_cases()
_EXPECTED_RESULTS = load_expected_results()


def _build_test_params() -> list[pytest.param]:
    params: list[pytest.param] = []
    tests = _EXPECTED_RESULTS["tests"]
    for case in _OFFICIAL_CASES:
        if case.case_id not in tests:
            raise AssertionError(
                f"Missing expected result entry for official test {case.case_id!r}"
            )
        expected_result = tests[case.case_id]["expected"]["comparison_result"]
        params.append(pytest.param(case, expected_result, id=case.case_id))
    return params


_TEST_PARAMS = _build_test_params()


def test_expected_results_cover_all_official_cases() -> None:
    case_ids = {case.case_id for case in _OFFICIAL_CASES}
    expected_ids = set(_EXPECTED_RESULTS["tests"])

    assert _EXPECTED_RESULTS["suite"]["case_count"] == len(_OFFICIAL_CASES)
    assert case_ids == expected_ids
    assert len(_OFFICIAL_CASES) >= 1800
    assert EXPECTED_RESULTS_PATH.exists()


@pytest.mark.parametrize("case, expected_result", _TEST_PARAMS)
def test_oscl_vs_onnx(case: Any, expected_result: str) -> None:
    actual_result = compare_case(case)
    assert actual_result == expected_result


# ---------------------------------------------------------------------------
# Direct engine unit tests (no ONNX test-case dependency)
# ---------------------------------------------------------------------------


class TestEngineBasic:
    """Unit tests for the OTSL engine on hand-crafted models."""

    @staticmethod
    def _simple_model(
        op_type: str,
        input_shapes: list[list[int]],
        output_names: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        initializers: list[tuple[str, np.ndarray]] | None = None,
    ) -> ModelProto:
        input_infos = []
        input_names = []
        for i, shape in enumerate(input_shapes):
            name = f"input_{i}"
            input_names.append(name)
            input_infos.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            )

        if output_names is None:
            output_names = ["output"]
        output_infos = [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, None)
            for n in output_names
        ]

        kwargs: dict[str, Any] = {}
        if attrs:
            kwargs.update(attrs)
        node = helper.make_node(op_type, input_names, output_names, **kwargs)
        inits = []
        if initializers:
            for init_name, arr in initializers:
                inits.append(numpy_helper.from_array(arr, name=init_name))
        graph = helper.make_graph(
            [node], "test_graph", input_infos, output_infos, initializer=inits
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )

    def test_add(self) -> None:
        m = self._simple_model("Add", [[3, 4, 5], [3, 4, 5]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_add_broadcast(self) -> None:
        m = self._simple_model("Add", [[3, 4, 5], [5]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_relu(self) -> None:
        m = self._simple_model("Relu", [[2, 3, 4]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [2, 3, 4]}

    def test_split_without_num_outputs_uses_node_output_count(self) -> None:
        m = self._simple_model("Split", [[2, 4]], output_names=["left", "right"], attrs={"axis": 1})
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"left": [2, 2], "right": [2, 2]}

    def test_matmul_2d(self) -> None:
        m = self._simple_model("MatMul", [[3, 4], [4, 5]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 5]}

    def test_matmul_4d_broadcast(self) -> None:
        m = self._simple_model("MatMul", [[3, 1, 3, 4], [1, 2, 4, 2]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 2, 3, 2]}

    def test_transpose_default_perm(self) -> None:
        m = self._simple_model("Transpose", [[2, 3, 4]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [4, 3, 2]}

    def test_transpose_explicit_perm(self) -> None:
        m = self._simple_model(
            "Transpose", [[2, 3, 4]], attrs={"perm": [1, 2, 0]}
        )
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 2]}

    def test_flatten_default(self) -> None:
        m = self._simple_model("Flatten", [[5, 4, 3, 2]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [5, 24]}

    def test_flatten_axis0(self) -> None:
        m = self._simple_model("Flatten", [[2, 3, 4, 5]], attrs={"axis": 0})
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [1, 120]}

    def test_gemm_no_trans(self) -> None:
        m = self._simple_model("Gemm", [[3, 5], [5, 4], [1, 4]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4]}

    def test_gemm_transA(self) -> None:
        m = self._simple_model(
            "Gemm", [[6, 3], [6, 4], [1, 4]], attrs={"transA": 1}
        )
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4]}

    def test_reshape(self) -> None:
        target = np.array([6, 4], dtype=np.int64)
        m = self._simple_model(
            "Reshape",
            [[2, 3, 4], [2]],
            initializers=[("input_1", target)],
        )
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [6, 4]}

    def test_reshape_neg1(self) -> None:
        target = np.array([2, -1], dtype=np.int64)
        m = self._simple_model(
            "Reshape",
            [[2, 3, 4], [2]],
            initializers=[("input_1", target)],
        )
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [2, 12]}

    def test_softmax(self) -> None:
        m = self._simple_model("Softmax", [[1, 3]])
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [1, 3]}

    def test_gather(self) -> None:
        m = self._simple_model(
            "Gather", [[5, 4, 3, 2], [3]], attrs={"axis": 0}
        )
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 3, 2]}

    def test_squeeze(self) -> None:
        axes = np.array([0], dtype=np.int64)
        m = self._simple_model(
            "Squeeze",
            [[1, 3, 4, 5], [1]],
            initializers=[("input_1", axes)],
        )
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_unsqueeze(self) -> None:
        axes = np.array([0], dtype=np.int64)
        m = self._simple_model(
            "Unsqueeze",
            [[3, 4, 5], [1]],
            initializers=[("input_1", axes)],
        )
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [1, 3, 4, 5]}

    def test_nonzero_rank(self) -> None:
        """NonZero: first dim = rank(X), second dim is unknown."""
        m = self._simple_model("NonZero", [[2, 2]])
        result = oscl_infer_shapes(m)
        shapes = get_output_shapes(result)
        assert shapes["output"][0] == 2

    def test_concat(self) -> None:
        m = self._simple_model("Concat", [[2, 3], [2, 4]], attrs={"axis": 1})
        result = oscl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [2, 7]}

    def test_interface_matches_onnx(self) -> None:
        """Verify the return type is ModelProto like ``onnx.shape_inference``."""
        m = self._simple_model("Relu", [[2, 3]])
        result = oscl_infer_shapes(m)
        assert isinstance(result, ModelProto)

    def test_missing_spec_raises(self) -> None:
        """Operators without bundled OTSL specs must fail explicitly."""
        m = self._typed_model(
            "MissingSpecOp",
            [([2, 3], TensorProto.FLOAT)],
            output_names=["y"],
        )
        with pytest.raises(
            NotImplementedError, match="MissingSpecOp"
        ):
            oscl_infer_shapes(m)

    @staticmethod
    def _typed_model(
        op_type: str,
        input_specs: list[tuple[list[int], int]],
        output_names: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        initializers: list[tuple[str, np.ndarray]] | None = None,
    ) -> ModelProto:
        """Create a model with explicit input element types."""
        input_infos = []
        input_names = []
        for i, (shape, elem_type) in enumerate(input_specs):
            name = f"input_{i}"
            input_names.append(name)
            input_infos.append(
                helper.make_tensor_value_info(name, elem_type, shape)
            )
        if output_names is None:
            output_names = ["output"]
        output_infos = [
            helper.make_tensor_value_info(n, TensorProto.UNDEFINED, None)
            for n in output_names
        ]
        kwargs: dict[str, Any] = {}
        if attrs:
            kwargs.update(attrs)
        node = helper.make_node(op_type, input_names, output_names, **kwargs)
        inits = []
        if initializers:
            for init_name, arr in initializers:
                inits.append(numpy_helper.from_array(arr, name=init_name))
        graph = helper.make_graph(
            [node], "test_graph", input_infos, output_infos, initializer=inits
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)]
        )

    def test_type_passthrough(self) -> None:
        """Unary op preserves input type."""
        m = self._typed_model("Relu", [([2, 3], TensorProto.DOUBLE)])
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.DOUBLE}

    def test_type_broadcast(self) -> None:
        """Binary broadcast op preserves first input type."""
        m = self._typed_model("Add", [
            ([3, 4], TensorProto.FLOAT16),
            ([3, 4], TensorProto.FLOAT16),
        ])
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.FLOAT16}

    def test_type_comparison_bool(self) -> None:
        """Comparison op produces bool."""
        m = self._typed_model("Equal", [
            ([3, 4], TensorProto.INT64),
            ([3, 4], TensorProto.INT64),
        ])
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.BOOL}

    def test_type_argmax_int64(self) -> None:
        """ArgMax produces int64."""
        m = self._typed_model("ArgMax", [([3, 4], TensorProto.FLOAT)])
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_type_cast(self) -> None:
        """Cast changes type per 'to' attribute."""
        m = self._typed_model(
            "Cast",
            [([2, 3], TensorProto.FLOAT)],
            attrs={"to": TensorProto.INT64},
        )
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_type_shape_int64(self) -> None:
        """Shape op always returns int64."""
        m = self._typed_model("Shape", [([2, 3, 4], TensorProto.FLOAT)])
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_type_nonzero_int64(self) -> None:
        """NonZero always returns int64."""
        m = self._typed_model("NonZero", [([2, 2], TensorProto.DOUBLE)])
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_type_concat_passthrough(self) -> None:
        """Concat preserves input type."""
        m = self._typed_model(
            "Concat",
            [([2, 3], TensorProto.DOUBLE), ([2, 4], TensorProto.DOUBLE)],
            attrs={"axis": 1},
        )
        result = oscl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.DOUBLE}
