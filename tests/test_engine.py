"""Parametrized tests comparing the OSCL shape inference engine against the
official ONNX shape inference for all official ONNX node-level test cases
of the operators covered by bundled OSCL specifications.

Each test case:
1. Collects the ONNX backend node test for a given operator.
2. Enriches the model with constant-tensor initializers so that both engines
   have equivalent information.
3. Runs the official ``onnx.shape_inference.infer_shapes``.
4. Runs the OSCL-based ``oscl.engine.infer_shapes``.
5. Asserts the inferred output shapes are identical.
"""

from __future__ import annotations

import copy
import importlib
import os
import pkgutil
import sys
import warnings
from typing import Any

import numpy as np
import onnx
import onnx.backend.test.case.node as _onnx_node_mod
import onnx.defs
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper, shape_inference

from oscl.engine import OsclShapeInferenceEngine, infer_shapes as oscl_infer_shapes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Suppress numpy overflow warnings emitted during ONNX test-case collection
# (e.g. cast.py, castlike.py).  Scoped to numpy's RuntimeWarnings only.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\.")

# Singleton engine (avoids re-parsing specs for every test case).
_ENGINE = OsclShapeInferenceEngine()

# ---------------------------------------------------------------------------
# Auto-discovery: ONNX op_type ↔ spec name ↔ ONNX test module name
# ---------------------------------------------------------------------------

# Reverse lookup: spec name (lowercase) → ONNX op_type (CamelCase), built
# from the ONNX operator schema registry.
_SPEC_TO_OP_TYPE: dict[str, str] = {}
for _schema in onnx.defs.get_all_schemas_with_history():
    _SPEC_TO_OP_TYPE[_schema.name.lower()] = _schema.name

# Set of available ONNX node-test modules (discovered once at import time).
_ONNX_TEST_MODULES: set[str] = {
    name
    for _, name, _ in pkgutil.iter_modules(
        [os.path.dirname(_onnx_node_mod.__file__)]
    )
}

# The ONNX test module name *usually* equals ``op_type.lower()`` but a few
# operators deviate.  This small table maps ONNX op_type to the actual ONNX
# backend test module name where the two differ (e.g. "Range" uses "rangeop"
# because "range" conflicts with the Python built-in).
_ONNX_TEST_MODULE_OVERRIDES: dict[str, str] = {
    "BatchNormalization": "batchnorm",
    "GreaterOrEqual": "greater_equal",
    "InstanceNormalization": "instancenorm",
    "LessOrEqual": "less_equal",
    "Range": "rangeop",
    "ReduceLogSum": "reduce_log_sum",
    "ReduceLogSumExp": "reduce_log_sum_exp",
    "SoftmaxCrossEntropyLoss": "softmaxcrossentropy",
}


def _onnx_test_module_name(op_type: str) -> str | None:
    """Return the ONNX backend test module name for *op_type*, or ``None``."""
    if op_type in _ONNX_TEST_MODULE_OVERRIDES:
        return _ONNX_TEST_MODULE_OVERRIDES[op_type]
    candidate = op_type.lower()
    if candidate in _ONNX_TEST_MODULES:
        return candidate
    return None


def _collect_onnx_node_tests(op_type: str, mod_name: str) -> list[Any]:
    """Import and collect the official ONNX node test cases for *op_type*."""
    import onnx.backend.test.case.node as nmod

    nmod._NodeTestCases = []
    nmod._TargetOpType = op_type

    mod_path = f"onnx.backend.test.case.node.{mod_name}"
    # Force re-import so the global target is respected.
    if mod_path in sys.modules:
        del sys.modules[mod_path]
    importlib.import_module(mod_path)
    return list(nmod._NodeTestCases)


def _inject_constant_inputs(
    model: ModelProto,
    input_arrays: list[np.ndarray],
) -> ModelProto:
    """Return a copy of *model* with input tensors added as initializers.

    Operators like Reshape, Squeeze and Unsqueeze read shape-valued inputs at
    inference time.  The ONNX test cases provide these values only via
    ``data_sets``; to allow *static* shape inference we inject them as
    graph initializers.
    """
    # Maximum number of elements in a float tensor to inject as initializer.
    # Shape-relevant float tensors (scales, sizes, etc.) are typically small;
    # large data tensors may contain NaN/Inf test values we don't want.
    _MAX_FLOAT_INIT_ELEMS = 16

    model = copy.deepcopy(model)
    graph = model.graph

    graph_inputs = list(graph.input)
    existing_init_names = {init.name for init in graph.initializer}

    for idx, inp in enumerate(graph_inputs):
        if idx >= len(input_arrays):
            break
        arr = input_arrays[idx]
        name = inp.name

        # Convert numpy scalars to 0-D arrays
        if isinstance(arr, np.generic) and not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # Skip non-numpy arrays (e.g., onnx custom dtype wrappers)
        if not isinstance(arr, np.ndarray):
            continue

        if name in existing_init_names:
            continue
        if arr.dtype.kind in ("i", "u"):
            tensor = numpy_helper.from_array(arr, name=name)
            graph.initializer.append(tensor)
        elif arr.dtype.kind == "f" and arr.size <= _MAX_FLOAT_INIT_ELEMS:
            # Inject small float tensors (scalars and short vectors) needed by
            # operators like Resize/Upsample (scales), Range (start/limit/delta).
            tensor = numpy_helper.from_array(arr, name=name)
            graph.initializer.append(tensor)

    return model


def _get_output_shapes(model: ModelProto) -> dict[str, list[int]]:
    """Extract output shapes from graph.output as ``{name: [dims]}``."""
    result: dict[str, list[int]] = {}
    for out in model.graph.output:
        tp = out.type.tensor_type
        if tp.HasField("shape"):
            result[out.name] = [d.dim_value for d in tp.shape.dim]
    return result


# ---------------------------------------------------------------------------
# Test-case collection
# ---------------------------------------------------------------------------

# Known-limitation test names where the OSCL spec cannot match ONNX due to
# data-dependent output dimensions not covered by the spec.
_XFAIL_CASES: set[str] = {
    # NonZero: second output dimension is data-dependent (unknown_nonnegative).
    "test_nonzero_example",
    # Einsum: ellipsis / scalar cases not fully supported yet.
    "test_einsum_batch_diagonal",
    "test_einsum_inner_prod",
    # Compress: output dimension is data-dependent (depends on condition values).
    "test_compress_0",
    "test_compress_1",
    "test_compress_default_axis",
    "test_compress_negative_axis",
    # NonMaxSuppression: first output dim is data-dependent (selected boxes).
    "test_nonmaxsuppression_suppress_by_IOU",
    "test_nonmaxsuppression_suppress_by_IOU_and_scores",
    "test_nonmaxsuppression_flipped_coordinates",
    "test_nonmaxsuppression_limit_output_size",
    "test_nonmaxsuppression_single_box",
    "test_nonmaxsuppression_identical_boxes",
    "test_nonmaxsuppression_center_point_box_format",
    "test_nonmaxsuppression_two_classes",
    "test_nonmaxsuppression_two_batches",
    # Unique: output dimension is data-dependent (unique element count).
    "test_unique_sorted_without_axis",
    "test_unique_not_sorted_without_axis",
    "test_unique_sorted_with_axis",
    "test_unique_sorted_with_axis_3d",
    "test_unique_sorted_with_negative_axis",
    "test_unique_length_1",
    # Resize with axes attribute or keep_aspect_ratio_policy not yet supported.
    "test_resize_upsample_scales_nearest_axes_2_3",
    "test_resize_upsample_scales_nearest_axes_3_2",
    "test_resize_upsample_sizes_nearest_axes_2_3",
    "test_resize_upsample_sizes_nearest_axes_3_2",
    "test_resize_tf_crop_and_resize_axes_2_3",
    "test_resize_tf_crop_and_resize_axes_3_2",
    "test_resize_upsample_sizes_nearest_not_larger",
    "test_resize_upsample_sizes_nearest_not_smaller",
    "test_resize_downsample_sizes_nearest_not_larger",
    "test_resize_downsample_sizes_nearest_not_smaller",
}


def _build_test_params() -> list[pytest.param]:
    """Collect parametrized test entries for every operator that has both an
    OSCL spec file and an ONNX backend test module."""
    params: list[pytest.param] = []
    for spec_name in sorted(_ENGINE._specs):
        op_type = _SPEC_TO_OP_TYPE.get(spec_name)
        if op_type is None:
            continue  # no matching ONNX schema
        mod_name = _onnx_test_module_name(op_type)
        if mod_name is None:
            continue  # no ONNX test module available
        cases = _collect_onnx_node_tests(op_type, mod_name)
        for tc in cases:
            # Some test cases use expanded sub-graphs with many nodes; we only
            # keep single-node tests whose op_type matches.
            if not any(n.op_type == op_type for n in tc.model.graph.node):
                continue
            test_id = tc.name
            marks: list[Any] = []
            if test_id in _XFAIL_CASES:
                marks.append(
                    pytest.mark.xfail(
                        reason="Known OSCL spec limitation", strict=True
                    )
                )
            params.append(pytest.param(tc, op_type, id=test_id, marks=marks))
    return params


_TEST_PARAMS = _build_test_params()


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("test_case, op_type", _TEST_PARAMS)
def test_oscl_vs_onnx(test_case: Any, op_type: str) -> None:
    """Compare OSCL engine output shapes against ONNX official inference."""
    model = test_case.model
    data_set = test_case.data_sets[0]
    input_arrays: list[np.ndarray] = list(data_set[0])
    expected_output_arrays: list[np.ndarray] = list(data_set[1])

    # Inject constant integer inputs (shape tensors, axes, etc.)
    enriched_model = _inject_constant_inputs(model, input_arrays)

    # Reference shapes come from the test model's output type info,
    # which the ONNX test framework pre-populates with correct shapes.
    onnx_inferred = shape_inference.infer_shapes(enriched_model)
    onnx_shapes = _get_output_shapes(onnx_inferred)

    # OSCL engine inference
    oscl_inferred = _ENGINE.infer_shapes(enriched_model)
    oscl_shapes = _get_output_shapes(oscl_inferred)

    # Compare every output
    for out_name, onnx_shape in onnx_shapes.items():
        oscl_shape = oscl_shapes.get(out_name)
        assert oscl_shape is not None, (
            f"OSCL engine did not produce shape for output {out_name!r}"
        )
        assert oscl_shape == onnx_shape, (
            f"[{op_type}/{test_case.name}] output {out_name!r}: "
            f"OSCL={oscl_shape} vs ONNX={onnx_shape}"
        )


# ---------------------------------------------------------------------------
# Sanity: ensure we actually collected a reasonable number of test cases
# ---------------------------------------------------------------------------


def test_collected_test_count() -> None:
    """We must have at least one test case per spec that has an ONNX test module."""
    ops_seen: set[str] = set()
    for p in _TEST_PARAMS:
        # The op_type is the second positional value in the param.
        ops_seen.add(p.values[1])
    for spec_name in sorted(_ENGINE._specs):
        op_type = _SPEC_TO_OP_TYPE.get(spec_name)
        if op_type is None:
            continue
        mod_name = _onnx_test_module_name(op_type)
        if mod_name is None:
            continue
        assert op_type in ops_seen, f"No test cases collected for {op_type}"


# ---------------------------------------------------------------------------
# Direct engine unit tests (no ONNX test-case dependency)
# ---------------------------------------------------------------------------


class TestEngineBasic:
    """Unit tests for the OSCL engine on hand-crafted models."""

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
        assert _get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_add_broadcast(self) -> None:
        m = self._simple_model("Add", [[3, 4, 5], [5]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_relu(self) -> None:
        m = self._simple_model("Relu", [[2, 3, 4]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [2, 3, 4]}

    def test_matmul_2d(self) -> None:
        m = self._simple_model("MatMul", [[3, 4], [4, 5]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 5]}

    def test_matmul_4d_broadcast(self) -> None:
        m = self._simple_model("MatMul", [[3, 1, 3, 4], [1, 2, 4, 2]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 2, 3, 2]}

    def test_transpose_default_perm(self) -> None:
        m = self._simple_model("Transpose", [[2, 3, 4]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [4, 3, 2]}

    def test_transpose_explicit_perm(self) -> None:
        m = self._simple_model(
            "Transpose", [[2, 3, 4]], attrs={"perm": [1, 2, 0]}
        )
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 4, 2]}

    def test_flatten_default(self) -> None:
        m = self._simple_model("Flatten", [[5, 4, 3, 2]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [5, 24]}

    def test_flatten_axis0(self) -> None:
        m = self._simple_model("Flatten", [[2, 3, 4, 5]], attrs={"axis": 0})
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [1, 120]}

    def test_gemm_no_trans(self) -> None:
        m = self._simple_model("Gemm", [[3, 5], [5, 4], [1, 4]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 4]}

    def test_gemm_transA(self) -> None:
        m = self._simple_model(
            "Gemm", [[6, 3], [6, 4], [1, 4]], attrs={"transA": 1}
        )
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 4]}

    def test_reshape(self) -> None:
        target = np.array([6, 4], dtype=np.int64)
        m = self._simple_model(
            "Reshape",
            [[2, 3, 4], [2]],
            initializers=[("input_1", target)],
        )
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [6, 4]}

    def test_reshape_neg1(self) -> None:
        target = np.array([2, -1], dtype=np.int64)
        m = self._simple_model(
            "Reshape",
            [[2, 3, 4], [2]],
            initializers=[("input_1", target)],
        )
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [2, 12]}

    def test_softmax(self) -> None:
        m = self._simple_model("Softmax", [[1, 3]])
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [1, 3]}

    def test_gather(self) -> None:
        m = self._simple_model(
            "Gather", [[5, 4, 3, 2], [3]], attrs={"axis": 0}
        )
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 4, 3, 2]}

    def test_squeeze(self) -> None:
        axes = np.array([0], dtype=np.int64)
        m = self._simple_model(
            "Squeeze",
            [[1, 3, 4, 5], [1]],
            initializers=[("input_1", axes)],
        )
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [3, 4, 5]}

    def test_unsqueeze(self) -> None:
        axes = np.array([0], dtype=np.int64)
        m = self._simple_model(
            "Unsqueeze",
            [[3, 4, 5], [1]],
            initializers=[("input_1", axes)],
        )
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [1, 3, 4, 5]}

    def test_nonzero_rank(self) -> None:
        """NonZero: first dim = rank(X), second dim is unknown."""
        m = self._simple_model("NonZero", [[2, 2]])
        result = oscl_infer_shapes(m)
        shapes = _get_output_shapes(result)
        # First dim must equal input rank
        assert shapes["output"][0] == 2

    def test_concat(self) -> None:
        m = self._simple_model("Concat", [[2, 3], [2, 4]], attrs={"axis": 1})
        result = oscl_infer_shapes(m)
        assert _get_output_shapes(result) == {"output": [2, 7]}

    def test_interface_matches_onnx(self) -> None:
        """Verify the return type is ModelProto like ``onnx.shape_inference``."""
        m = self._simple_model("Relu", [[2, 3]])
        result = oscl_infer_shapes(m)
        assert isinstance(result, ModelProto)
