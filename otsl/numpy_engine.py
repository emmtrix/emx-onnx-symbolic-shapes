"""OTSL-based numpy shape inference engine for ONNX models.

This module provides an :func:`infer_shapes` function whose interface mirrors
``onnx.shape_inference.infer_shapes``.  Unlike the numerical engine which works
with plain Python ints, this engine represents shape dimensions as numpy
scalars (``np.int64``) so that all internal arithmetic is performed by numpy.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, TypeProto, helper, numpy_helper

from .loader import load_all_specs
from .numerical_engine import (
    ConstraintViolation,
    DimValue,
    _DEFAULT_ATTRS,
    _execute_spec,
    _get_attribute_value,
    _get_elem_type,
    _get_shape_from_type,
    _get_tensor_values,
    _infer_tensor_value_output,
    _is_known_int,
    _make_type_proto,
    _merge_or_set_shape,
    _merge_or_set_type,
    _merge_type_protos,
)

__all__ = ["infer_shapes", "OtslNumpyShapeInferenceEngine"]


def _to_numpy_shape(shape: list[DimValue]) -> list[Any]:
    """Convert a shape list so that concrete integer dims are ``np.int64``."""
    return [
        np.int64(d) if isinstance(d, int) and not isinstance(d, bool) else d
        for d in shape
    ]


def _numpy_int_values(values: list[Any]) -> list[Any]:
    """Convert integer values in a list to ``np.int64``."""
    return [
        np.int64(v) if isinstance(v, int) and not isinstance(v, bool) else v
        for v in values
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OtslNumpyShapeInferenceEngine:
    """Numpy-based shape inference engine backed by OTSL operator specifications.

    Identical in interface to
    :class:`~otsl.numerical_engine.OtslNumericalShapeInferenceEngine` but
    dimensions are represented as ``np.int64`` scalars internally so that all
    shape arithmetic is carried out by numpy rather than plain Python.

    Usage::

        engine = OtslNumpyShapeInferenceEngine()
        inferred_model = engine.infer_shapes(model)
    """

    def __init__(self) -> None:
        self._specs = load_all_specs()

    # -----------------------------------------------------------------------
    # Shape inference (mirrors numerical_engine but with numpy dimensions)
    # -----------------------------------------------------------------------

    def infer_shapes(self, model: ModelProto) -> ModelProto:
        """Infer shapes for all nodes in *model*.

        The interface mirrors ``onnx.shape_inference.infer_shapes``:
        returns a *new* ``ModelProto`` with inferred shapes populated in
        ``graph.value_info`` and ``graph.output``.
        """
        model = copy.deepcopy(model)
        graph = model.graph

        # Collect known shapes ------------------------------------------------
        known_shapes: dict[str, list[Any]] = {}
        known_elem_types: dict[str, int] = {}
        known_types: dict[str, TypeProto] = {}

        for inp in graph.input:
            s = _get_shape_from_type(inp.type)
            if s is not None:
                known_shapes[inp.name] = _to_numpy_shape(s)
            known_elem_types[inp.name] = _get_elem_type(inp.type)
            known_types[inp.name] = copy.deepcopy(inp.type)

        for vi in graph.value_info:
            s = _get_shape_from_type(vi.type)
            if s is not None:
                known_shapes[vi.name] = _to_numpy_shape(s)
            known_elem_types[vi.name] = _get_elem_type(vi.type)
            known_types[vi.name] = copy.deepcopy(vi.type)

        for out in graph.output:
            known_types.setdefault(out.name, copy.deepcopy(out.type))

        # Initialiser shapes and values ---------------------------------------
        initializer_values: dict[str, list[Any]] = {}
        known_tensor_values: dict[str, list[Any]] = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            known_shapes[init.name] = _to_numpy_shape(list(arr.shape))
            known_elem_types.setdefault(init.name, init.data_type)
            known_types.setdefault(
                init.name,
                _make_type_proto(list(arr.shape), init.data_type),
            )
            if init.data_type in (
                TensorProto.INT32,
                TensorProto.INT64,
                TensorProto.INT16,
                TensorProto.INT8,
            ):
                initializer_values[init.name] = _numpy_int_values(
                    arr.flatten().astype(int).tolist()
                )
            elif init.data_type in (
                TensorProto.FLOAT,
                TensorProto.DOUBLE,
                TensorProto.FLOAT16,
            ):
                flat = arr.flatten().tolist()
                if any(math.isnan(v) or math.isinf(v) for v in flat):
                    continue
                initializer_values[init.name] = [
                    np.int64(int(v)) if v == int(v) else v for v in flat
                ]
            tensor_values = _get_tensor_values(init)
            if tensor_values is not None:
                known_tensor_values[init.name] = _numpy_int_values(tensor_values)

        # Forward pass: infer shapes node by node -----------------------------
        value_info_names = {vi.name for vi in graph.value_info}

        for node in graph.node:
            spec_name = node.op_type.lower()
            if spec_name not in self._specs:
                if node.domain:
                    raise NotImplementedError(
                        "No OTSL spec available for operator "
                        f"{node.op_type!r} in domain {node.domain!r}"
                    )
                raise NotImplementedError(
                    f"No OTSL spec available for operator {node.op_type!r}"
                )

            spec = self._specs[spec_name]

            # Map ONNX node inputs -> OTSL spec input names
            input_shapes: dict[str, list[Any]] = {}
            input_elem_types: dict[str, Any] = {}
            input_full_types: dict[str, TypeProto] = {
                name: copy.deepcopy(tp) for name, tp in known_types.items()
            }
            tensor_vals: dict[str, list[Any]] = {}

            if spec.inputs and spec.inputs[0].variadic:
                var_name = spec.inputs[0].name
                var_shapes: list[Any] = []
                var_types: list[int] = []
                for onnx_in in node.input:
                    if onnx_in and onnx_in in known_shapes:
                        var_shapes.append(known_shapes[onnx_in])
                    else:
                        var_shapes.append([])
                    if onnx_in:
                        var_types.append(
                            known_elem_types.get(onnx_in, TensorProto.UNDEFINED)
                        )
                    else:
                        var_types.append(TensorProto.UNDEFINED)
                input_shapes[var_name] = var_shapes  # type: ignore[assignment]
                input_elem_types[var_name] = var_types
            else:
                for i, inp_decl in enumerate(spec.inputs):
                    if i < len(node.input) and node.input[i]:
                        onnx_name = node.input[i]
                        if onnx_name in known_shapes:
                            input_shapes[inp_decl.name] = known_shapes[onnx_name]
                        elif onnx_name in known_types:
                            shape = _get_shape_from_type(known_types[onnx_name])
                            if shape is not None:
                                input_shapes[inp_decl.name] = _to_numpy_shape(shape)
                        if onnx_name in known_elem_types:
                            input_elem_types[inp_decl.name] = known_elem_types[onnx_name]
                        elif onnx_name in known_types:
                            elem_type = _get_elem_type(known_types[onnx_name])
                            if elem_type != TensorProto.UNDEFINED:
                                input_elem_types[inp_decl.name] = elem_type
                        if onnx_name in known_types:
                            input_full_types[inp_decl.name] = copy.deepcopy(
                                known_types[onnx_name]
                            )
                        if onnx_name in known_tensor_values:
                            tensor_vals[inp_decl.name] = list(
                                known_tensor_values[onnx_name]
                            )

            # Map node attributes -> OTSL attribute names
            attrs: dict[str, Any] = {}
            if node.op_type in _DEFAULT_ATTRS:
                attrs.update(_DEFAULT_ATTRS[node.op_type])
            node_attr_map = {a.name: a for a in node.attribute}
            for attr_name, attr in node_attr_map.items():
                attrs[attr_name] = _get_attribute_value(attr)
            for attr_name in spec.attributes:
                if attr_name in node_attr_map:
                    attrs[attr_name] = _get_attribute_value(node_attr_map[attr_name])

            # Execute the spec -------------------------------------------------
            try:
                node_input_types = [
                    copy.deepcopy(known_types[name]) if name in known_types else None
                    for name in node.input
                ]
                node_output_types = [
                    copy.deepcopy(known_types[name]) if name in known_types else None
                    for name in node.output
                ]
                output_shapes, output_types, output_values, output_onnx_types = (
                    _execute_spec(
                        spec,
                        input_shapes,
                        attrs,
                        tensor_vals,
                        elem_types=input_elem_types,
                        attribute_protos=node_attr_map,
                        full_types=input_full_types,
                        node_input_types=node_input_types,
                        node_output_types=node_output_types,
                    )
                )
            except (
                ConstraintViolation,
                ValueError,
                TypeError,
                IndexError,
                KeyError,
                NameError,
                ZeroDivisionError,
                AttributeError,
            ):
                continue  # graceful degradation

            # Store results back -----------------------------------------------
            if len(spec.outputs) == 1:
                out_name = spec.outputs[0]
                inferred_onnx_many = output_onnx_types.get(out_name)
                if (
                    isinstance(inferred_onnx_many, list)
                    and inferred_onnx_many
                    and all(isinstance(tp, TypeProto) for tp in inferred_onnx_many)
                ):
                    for onnx_out, output_tp in zip(node.output, inferred_onnx_many):
                        if not onnx_out:
                            continue
                        tp = copy.deepcopy(output_tp)
                        _merge_or_set_type(known_types, onnx_out, tp)
                        shape = _get_shape_from_type(tp)
                        if shape is not None:
                            _merge_or_set_shape(known_shapes, onnx_out, shape)
                        elem_type = _get_elem_type(tp)
                        if elem_type != TensorProto.UNDEFINED:
                            known_elem_types[onnx_out] = elem_type
                        if onnx_out not in value_info_names:
                            vi = graph.value_info.add()
                            vi.name = onnx_out
                            vi.type.CopyFrom(tp)
                            value_info_names.add(onnx_out)
                    continue

            if len(spec.outputs) == 1:
                out_name = spec.outputs[0]
                inferred_many = output_shapes.get(out_name)
                if (
                    isinstance(inferred_many, list)
                    and inferred_many
                    and all(isinstance(shape, (list, tuple)) for shape in inferred_many)
                ):
                    et = output_types.get(out_name, TensorProto.UNDEFINED)
                    if et == TensorProto.UNDEFINED and node.input:
                        et = known_elem_types.get(
                            node.input[0], TensorProto.UNDEFINED
                        )
                    for onnx_out, inferred in zip(node.output, inferred_many):
                        if not onnx_out:
                            continue
                        shape = list(inferred)
                        _merge_or_set_shape(known_shapes, onnx_out, shape)
                        known_elem_types[onnx_out] = et
                        tp = _make_type_proto(shape, et)
                        _merge_or_set_type(known_types, onnx_out, tp)
                        if onnx_out not in value_info_names:
                            vi = graph.value_info.add()
                            vi.name = onnx_out
                            vi.type.CopyFrom(tp)
                            value_info_names.add(onnx_out)
                    continue

            for j, out_name in enumerate(spec.outputs):
                if j < len(node.output) and node.output[j]:
                    onnx_out = node.output[j]
                    if out_name in output_onnx_types:
                        output_tp = copy.deepcopy(output_onnx_types[out_name])
                        _merge_or_set_type(known_types, onnx_out, output_tp)
                        shape = _get_shape_from_type(output_tp)
                        if shape is not None:
                            _merge_or_set_shape(known_shapes, onnx_out, shape)
                        elem_type = _get_elem_type(output_tp)
                        if elem_type != TensorProto.UNDEFINED:
                            known_elem_types[onnx_out] = elem_type
                        if onnx_out not in value_info_names:
                            vi = graph.value_info.add()
                            vi.name = onnx_out
                            vi.type.CopyFrom(output_tp)
                            value_info_names.add(onnx_out)

                    inferred = output_shapes.get(out_name)
                    if inferred is not None:
                        _merge_or_set_shape(known_shapes, onnx_out, inferred)

                        et = output_types.get(out_name, TensorProto.UNDEFINED)
                        if et == TensorProto.UNDEFINED and node.input:
                            et = known_elem_types.get(
                                node.input[0], TensorProto.UNDEFINED
                            )
                        known_elem_types[onnx_out] = et

                        tp = _make_type_proto(inferred, et)
                        _merge_or_set_type(known_types, onnx_out, tp)
                        if onnx_out not in value_info_names:
                            vi = graph.value_info.add()
                            vi.name = onnx_out
                            vi.type.CopyFrom(tp)
                            value_info_names.add(onnx_out)
                    if out_name in output_values:
                        known_tensor_values[onnx_out] = _numpy_int_values(
                            list(output_values[out_name])
                        )

                    try:
                        tensor_values = _infer_tensor_value_output(
                            node,
                            known_shapes,
                            known_tensor_values,
                            attrs,
                            node_attr_map,
                        )
                    except (TypeError, ValueError, IndexError, ZeroDivisionError):
                        tensor_values = None
                    if tensor_values is not None:
                        known_tensor_values[onnx_out] = _numpy_int_values(
                            tensor_values
                        )

        # Patch graph outputs with inferred shapes ----------------------------
        for out in graph.output:
            if out.name in known_types:
                out.type.CopyFrom(
                    _merge_type_protos(out.type, known_types[out.name])
                )
                continue
            if out.name in known_shapes:
                inferred = known_shapes[out.name]
                et = known_elem_types.get(out.name, _get_elem_type(out.type))
                out.type.CopyFrom(
                    _merge_type_protos(out.type, _make_type_proto(inferred, et))
                )

        return model

    # -----------------------------------------------------------------------
    # Special-case helpers (identical to numerical engine)
    # -----------------------------------------------------------------------

    @staticmethod
    def _handle_constant(
        node: Any,
        node_attr_map: dict[str, Any],
    ) -> tuple[list[Any] | None, int]:
        """Infer shape and element type of a Constant node from its attribute."""
        if "value" in node_attr_map:
            attr = node_attr_map["value"]
            if attr.type == onnx.AttributeProto.TENSOR:
                t = attr.t
                shape = [np.int64(d) for d in t.dims] if t.dims else []
                return shape, t.data_type
        if "value_int" in node_attr_map:
            return [], TensorProto.INT64
        if "value_float" in node_attr_map:
            return [], TensorProto.FLOAT
        if "value_ints" in node_attr_map:
            attr = node_attr_map["value_ints"]
            return [np.int64(len(attr.ints))], TensorProto.INT64
        if "value_floats" in node_attr_map:
            attr = node_attr_map["value_floats"]
            return [np.int64(len(attr.floats))], TensorProto.FLOAT
        if "value_string" in node_attr_map:
            return [], TensorProto.STRING
        if "value_strings" in node_attr_map:
            attr = node_attr_map["value_strings"]
            return [np.int64(len(attr.strings))], TensorProto.STRING
        return None, TensorProto.UNDEFINED


# Module-level convenience function matching ``onnx.shape_inference.infer_shapes``.
_DEFAULT_ENGINE: OtslNumpyShapeInferenceEngine | None = None


def infer_shapes(model: ModelProto) -> ModelProto:
    """Infer shapes for *model* using OTSL specifications with numpy arithmetic.

    Drop-in replacement for ``onnx.shape_inference.infer_shapes``.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = OtslNumpyShapeInferenceEngine()
    return _DEFAULT_ENGINE.infer_shapes(model)
