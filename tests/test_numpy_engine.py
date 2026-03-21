"""Tests for the OTSL numpy shape inference engine."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import ModelProto, TensorProto, TypeProto, helper, numpy_helper
from onnx import shape_inference

from otsl.numpy_engine import OtslNumpyShapeInferenceEngine
from otsl.numpy_engine import infer_shapes as otsl_infer_shapes
from tests.official_numerical_engine_suite import (
    OfficialTestCase,
    collect_official_test_cases,
)

# Suppress numpy overflow warnings emitted during ONNX test-case collection.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\.")

EXPECTED_RESULTS_PATH = (
    Path(__file__).resolve().parent / "data" / "test_numpy_engine_expected.json"
)


@dataclass(frozen=True)
class OfficialCaseExpectation:
    """One official test case paired with its frozen comparison result."""

    case: OfficialTestCase
    expected_comparison_result: str


def load_expected_results(path: Path = EXPECTED_RESULTS_PATH) -> dict[str, Any]:
    """Load the JSON document containing expected comparison results."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_case_expectations(
    cases: list[OfficialTestCase],
    expected_results: dict[str, Any],
) -> list[OfficialCaseExpectation]:
    """Pair each official test case with its expected comparison result."""
    tests = expected_results["tests"]
    expectations: list[OfficialCaseExpectation] = []
    for case in cases:
        if case.case_id not in tests:
            raise AssertionError(
                f"Missing expected result entry for official test {case.case_id!r}"
            )
        expectations.append(
            OfficialCaseExpectation(
                case=case,
                expected_comparison_result=tests[case.case_id]["expected"][
                    "comparison_result"
                ],
            )
        )
    return expectations


NUMPY_ENGINE = OtslNumpyShapeInferenceEngine()
_OFFICIAL_CASES = collect_official_test_cases()
if EXPECTED_RESULTS_PATH.exists():
    _EXPECTED_RESULTS = load_expected_results()
    _CASE_EXPECTATIONS = build_case_expectations(
        _OFFICIAL_CASES,
        _EXPECTED_RESULTS,
    )
else:
    _EXPECTED_RESULTS = {"suite": {"case_count": 0}, "tests": {}}
    _CASE_EXPECTATIONS = []
_OFFICIAL_CASE_PARAMS = [
    pytest.param(expectation, id=expectation.case.case_id)
    for expectation in _CASE_EXPECTATIONS
]


def test_expected_results_cover_all_official_cases() -> None:
    case_ids = {case.case_id for case in _OFFICIAL_CASES}
    expected_ids = set(_EXPECTED_RESULTS["tests"])

    assert _EXPECTED_RESULTS["suite"]["case_count"] == len(_OFFICIAL_CASES)
    assert case_ids == expected_ids
    assert len(_OFFICIAL_CASES) >= 1800
    assert EXPECTED_RESULTS_PATH.exists()


def compare_official_case_with_onnx(case: OfficialTestCase) -> str:
    """Return ``OK`` or a deterministic error string for one official test.

    Unlike the numerical-engine test, this comparison does **not** inject
    constant inputs into the model.  The numpy engine processes the model
    as-is, and concrete numbers are compared only at the end.
    """
    try:
        model = _load_model(case)
        # No injection – run both engines on the model directly.
        onnx_outputs = _get_output_signatures(
            shape_inference.infer_shapes(copy.deepcopy(model))
        )
        otsl_outputs = _get_output_signatures(
            NUMPY_ENGINE.infer_shapes(model)
        )
        return _first_output_mismatch(onnx_outputs, otsl_outputs) or "OK"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


@pytest.mark.parametrize("expectation", _OFFICIAL_CASE_PARAMS)
def test_otsl_numpy_vs_onnx(expectation: OfficialCaseExpectation) -> None:
    actual_result = compare_official_case_with_onnx(expectation.case)
    assert actual_result == expectation.expected_comparison_result, (
        f"{expectation.case.case_id}: "
        f"expected {expectation.expected_comparison_result!r}, "
        f"got {actual_result!r}"
    )


# ---------------------------------------------------------------------------
# Official ONNX comparison helpers
# ---------------------------------------------------------------------------


def _load_model(case: OfficialTestCase) -> ModelProto:
    return onnx.load(_resolve_model_path(case))


def _resolve_model_path(case: OfficialTestCase) -> Path:
    if case.model_dir is not None:
        return Path(case.model_dir) / "model.onnx"
    if case.url and case.url.startswith("onnx/backend/test/data/light/"):
        site_packages = Path(onnx.__file__).resolve().parent.parent
        return site_packages / case.url
    raise FileNotFoundError(
        f"Unable to resolve model path for official test {case.case_id!r}"
    )


def _first_output_mismatch(
    onnx_outputs: dict[str, dict[str, Any]],
    otsl_outputs: dict[str, dict[str, Any]],
) -> str | None:
    if set(onnx_outputs) != set(otsl_outputs):
        return (
            "output names differ: "
            f"OTSL={sorted(otsl_outputs)} ONNX={sorted(onnx_outputs)}"
        )

    for out_name in sorted(onnx_outputs):
        onnx_sig = onnx_outputs[out_name]
        otsl_sig = otsl_outputs[out_name]
        if otsl_sig != onnx_sig:
            return (
                f"output {out_name!r} mismatch: "
                f"OTSL={json.dumps(otsl_sig, sort_keys=True)} "
                f"ONNX={json.dumps(onnx_sig, sort_keys=True)}"
            )
    return None


def _get_output_signatures(model: ModelProto) -> dict[str, dict[str, Any]]:
    return {out.name: _type_signature(out.type) for out in model.graph.output}


def _type_signature(type_proto: TypeProto) -> dict[str, Any]:
    if type_proto.HasField("tensor_type"):
        tensor_type = type_proto.tensor_type
        signature: dict[str, Any] = {
            "kind": "tensor",
            "elem_type": TensorProto.DataType.Name(tensor_type.elem_type),
        }
        if tensor_type.HasField("shape"):
            signature["shape"] = [
                _dim_signature(dim) for dim in tensor_type.shape.dim
            ]
        return signature

    if type_proto.HasField("sequence_type"):
        return {
            "kind": "sequence",
            "elem": _type_signature(type_proto.sequence_type.elem_type),
        }

    if type_proto.HasField("optional_type"):
        return {
            "kind": "optional",
            "elem": _type_signature(type_proto.optional_type.elem_type),
        }

    if type_proto.HasField("map_type"):
        return {
            "kind": "map",
            "key_type": TensorProto.DataType.Name(type_proto.map_type.key_type),
            "value": _type_signature(type_proto.map_type.value_type),
        }

    if type_proto.HasField("sparse_tensor_type"):
        sparse_type = type_proto.sparse_tensor_type
        signature = {
            "kind": "sparse_tensor",
            "elem_type": TensorProto.DataType.Name(sparse_type.elem_type),
        }
        if sparse_type.HasField("shape"):
            signature["shape"] = [
                _dim_signature(dim) for dim in sparse_type.shape.dim
            ]
        return signature

    return {"kind": "unknown"}


def _dim_signature(dim: onnx.TensorShapeProto.Dimension) -> int | str:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return "?"


# ---------------------------------------------------------------------------
# Direct numpy engine unit-test helpers
# ---------------------------------------------------------------------------


def get_output_shapes(model: ModelProto) -> dict[str, list[int | str]]:
    """Extract tensor output shapes as ``{name: [dims]}``."""
    result: dict[str, list[int | str]] = {}
    for out in model.graph.output:
        if out.type.WhichOneof("value") == "tensor_type":
            tensor_type = out.type.tensor_type
            if tensor_type.HasField("shape"):
                result[out.name] = [_dim_signature(dim) for dim in tensor_type.shape.dim]
    return result


def get_output_types(model: ModelProto) -> dict[str, int]:
    """Extract tensor output element types as ``{name: elem_type}``."""
    result: dict[str, int] = {}
    for out in model.graph.output:
        if out.type.HasField("tensor_type"):
            elem_type = out.type.tensor_type.elem_type
            if elem_type != TensorProto.UNDEFINED:
                result[out.name] = elem_type
    return result


# ---------------------------------------------------------------------------
# Direct numpy engine unit tests (no ONNX test-case dependency)
# ---------------------------------------------------------------------------


class TestNumpyEngineBasic:
    """Unit tests for the OTSL numpy engine on hand-crafted models.

    These tests exercise the engine directly without injecting constant
    inputs; concrete numbers are compared only in the final assertions.
    """

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
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_add_broadcast(self) -> None:
        m = self._simple_model("Add", [[3, 4, 5], [5]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_relu(self) -> None:
        m = self._simple_model("Relu", [[2, 3, 4]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [2, 3, 4]}

    def test_relu_preserves_zero_dim(self) -> None:
        m = self._simple_model("Relu", [[2, 0, 4]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [2, 0, 4]}

    def test_split_without_num_outputs_uses_node_output_count(self) -> None:
        m = self._simple_model("Split", [[2, 4]], output_names=["left", "right"], attrs={"axis": 1})
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"left": [2, 2], "right": [2, 2]}

    def test_split_preserves_zero_sized_outputs(self) -> None:
        split = np.array([0, 3], dtype=np.int64)
        m = self._simple_model(
            "Split",
            [[3], [2]],
            output_names=["left", "right"],
            initializers=[("input_1", split)],
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"left": [0], "right": [3]}

    def test_matmul_2d(self) -> None:
        m = self._simple_model("MatMul", [[3, 4], [4, 5]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 5]}

    def test_matmul_4d_broadcast(self) -> None:
        m = self._simple_model("MatMul", [[3, 1, 3, 4], [1, 2, 4, 2]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 2, 3, 2]}

    def test_transpose_default_perm(self) -> None:
        m = self._simple_model("Transpose", [[2, 3, 4]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [4, 3, 2]}

    def test_transpose_explicit_perm(self) -> None:
        m = self._simple_model(
            "Transpose", [[2, 3, 4]], attrs={"perm": [1, 2, 0]}
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 2]}

    def test_flatten_default(self) -> None:
        m = self._simple_model("Flatten", [[5, 4, 3, 2]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [5, 24]}

    def test_flatten_axis0(self) -> None:
        m = self._simple_model("Flatten", [[2, 3, 4, 5]], attrs={"axis": 0})
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [1, 120]}

    def test_gemm_no_trans(self) -> None:
        m = self._simple_model("Gemm", [[3, 5], [5, 4], [1, 4]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4]}

    def test_gemm_transA(self) -> None:
        m = self._simple_model(
            "Gemm", [[6, 3], [6, 4], [1, 4]], attrs={"transA": 1}
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4]}

    def test_reshape(self) -> None:
        target = np.array([6, 4], dtype=np.int64)
        m = self._simple_model(
            "Reshape",
            [[2, 3, 4], [2]],
            initializers=[("input_1", target)],
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [6, 4]}

    def test_reshape_neg1(self) -> None:
        target = np.array([2, -1], dtype=np.int64)
        m = self._simple_model(
            "Reshape",
            [[2, 3, 4], [2]],
            initializers=[("input_1", target)],
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [2, 12]}

    def test_softmax(self) -> None:
        m = self._simple_model("Softmax", [[1, 3]])
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [1, 3]}

    def test_gather(self) -> None:
        m = self._simple_model(
            "Gather", [[5, 4, 3, 2], [3]], attrs={"axis": 0}
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 3, 2]}

    def test_squeeze(self) -> None:
        axes = np.array([0], dtype=np.int64)
        m = self._simple_model(
            "Squeeze",
            [[1, 3, 4, 5], [1]],
            initializers=[("input_1", axes)],
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_unsqueeze(self) -> None:
        axes = np.array([0], dtype=np.int64)
        m = self._simple_model(
            "Unsqueeze",
            [[3, 4, 5], [1]],
            initializers=[("input_1", axes)],
        )
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [1, 3, 4, 5]}

    def test_nonzero_rank(self) -> None:
        """NonZero: first dim = rank(X), second dim is unknown."""
        m = self._simple_model("NonZero", [[2, 2]])
        result = otsl_infer_shapes(m)
        shapes = get_output_shapes(result)
        assert shapes["output"][0] == 2

    def test_concat(self) -> None:
        m = self._simple_model("Concat", [[2, 3], [2, 4]], attrs={"axis": 1})
        result = otsl_infer_shapes(m)
        assert get_output_shapes(result) == {"output": [2, 7]}

    def test_interface_matches_onnx(self) -> None:
        """Verify the return type is ModelProto like ``onnx.shape_inference``."""
        m = self._simple_model("Relu", [[2, 3]])
        result = otsl_infer_shapes(m)
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
            otsl_infer_shapes(m)

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

    @staticmethod
    def _split_to_sequence_model(
        data_shape: list[int],
        split_shape: list[int] | None,
        *,
        axis: int,
        keepdims: int = 1,
    ) -> ModelProto:
        inputs = [helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)]
        node_inputs = ["data"]
        if split_shape is not None:
            inputs.append(
                helper.make_tensor_value_info("split", TensorProto.INT64, split_shape)
            )
            node_inputs.append("split")
        outputs = [
            helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, None)
        ]
        node = helper.make_node(
            "SplitToSequence", node_inputs, ["seq"], axis=axis, keepdims=keepdims
        )
        graph = helper.make_graph([node], "test_graph", inputs, outputs)
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )

    def test_type_passthrough(self) -> None:
        """Unary op preserves input type."""
        m = self._typed_model("Relu", [([2, 3], TensorProto.DOUBLE)])
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.DOUBLE}

    def test_type_broadcast(self) -> None:
        """Binary broadcast op preserves first input type."""
        m = self._typed_model("Add", [
            ([3, 4], TensorProto.FLOAT16),
            ([3, 4], TensorProto.FLOAT16),
        ])
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.FLOAT16}

    def test_type_comparison_bool(self) -> None:
        """Comparison op produces bool."""
        m = self._typed_model("Equal", [
            ([3, 4], TensorProto.INT64),
            ([3, 4], TensorProto.INT64),
        ])
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.BOOL}

    def test_type_argmax_int64(self) -> None:
        """ArgMax produces int64."""
        m = self._typed_model("ArgMax", [([3, 4], TensorProto.FLOAT)])
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_type_cast(self) -> None:
        """Cast changes type per 'to' attribute."""
        m = self._typed_model(
            "Cast",
            [([2, 3], TensorProto.FLOAT)],
            attrs={"to": TensorProto.INT64},
        )
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_type_shape_int64(self) -> None:
        """Shape op always returns int64."""
        m = self._typed_model("Shape", [([2, 3, 4], TensorProto.FLOAT)])
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_type_nonzero_int64(self) -> None:
        """NonZero always returns int64."""
        m = self._typed_model("NonZero", [([2, 2], TensorProto.DOUBLE)])
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.INT64}

    def test_split_to_sequence_scalar_split_without_value_keeps_unknown_dim(self) -> None:
        m = self._split_to_sequence_model([3, 6], [], axis=1)
        result = otsl_infer_shapes(m)
        signature = _type_signature(result.graph.output[0].type)
        assert signature["kind"] == "sequence"
        assert signature["elem"]["kind"] == "tensor"
        assert signature["elem"]["elem_type"] == "FLOAT"
        assert signature["elem"]["shape"][0] == 3
        assert isinstance(signature["elem"]["shape"][1], str)

    def test_split_to_sequence_vector_split_without_value_keeps_unknown_dim(self) -> None:
        m = self._split_to_sequence_model([3, 6], [2], axis=0)
        result = otsl_infer_shapes(m)
        signature = _type_signature(result.graph.output[0].type)
        assert signature["kind"] == "sequence"
        assert signature["elem"]["kind"] == "tensor"
        assert signature["elem"]["elem_type"] == "FLOAT"
        assert isinstance(signature["elem"]["shape"][0], str)
        assert signature["elem"]["shape"][1] == 6

    def test_split_to_sequence_without_split_input_uses_unit_dim(self) -> None:
        m = self._split_to_sequence_model([3, 6], None, axis=1)
        result = otsl_infer_shapes(m)
        signature = _type_signature(result.graph.output[0].type)
        assert signature["kind"] == "sequence"
        assert signature["elem"]["shape"] == [3, 1]

    def test_type_concat_passthrough(self) -> None:
        """Concat preserves input type."""
        m = self._typed_model(
            "Concat",
            [([2, 3], TensorProto.DOUBLE), ([2, 4], TensorProto.DOUBLE)],
            attrs={"axis": 1},
        )
        result = otsl_infer_shapes(m)
        assert get_output_types(result) == {"output": TensorProto.DOUBLE}
