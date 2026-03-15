"""OSCL-based shape inference engine for ONNX models.

This module provides an :func:`infer_shapes` function whose interface mirrors
``onnx.shape_inference.infer_shapes``.  Internally it evaluates the bundled
OSCL operator specifications to compute output shapes numerically.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, TypeProto, helper, numpy_helper

from .ast import (
    BinOp,
    Expr,
    FuncCall,
    Identifier,
    IfExpr,
    IndexExpr,
    LetStmt,
    MapExpr,
    NumberLit,
    RequireStmt,
    ResultStmt,
    ShapeLiteral,
    ShapeSpec,
    StringLit,
    UnknownDim,
    WhenStmt,
)
from .loader import load_all_specs

__all__ = ["infer_shapes", "OsclShapeInferenceEngine"]

# Sentinel for unknown dimensions (data-dependent).
_UNKNOWN = None


# ---------------------------------------------------------------------------
# Built-in OSCL functions
# ---------------------------------------------------------------------------


def _to_shape(val: Any) -> list[int | None]:
    """Coerce *val* to a shape (list of dimension ints or None)."""
    if isinstance(val, (list, tuple)):
        return list(val)
    raise TypeError(f"Cannot coerce {type(val).__name__} to shape: {val!r}")


def _builtin_shape(args: list[Any]) -> list[int | None]:
    """``shape(X)`` – return the shape of tensor *X*."""
    (val,) = args
    return _to_shape(val)


def _builtin_dim(args: list[Any]) -> int | None:
    """``dim(X, i)`` – return dimension *i* (supports negative indexing)."""
    shape, idx = args
    shape = _to_shape(shape)
    return shape[idx]


def _builtin_rank(args: list[Any]) -> int:
    """``rank(X)`` – number of dimensions."""
    (val,) = args
    return len(_to_shape(val))


def _builtin_prefix(args: list[Any]) -> list[int | None]:
    """``prefix(X, k)`` – first *k* dimensions (negative *k* ⇒ all but last |k|)."""
    shape, k = args
    shape = _to_shape(shape)
    if k < 0:
        end = len(shape) + k
        return shape[:max(end, 0)]
    return shape[:k]


def _builtin_suffix(args: list[Any]) -> list[int | None]:
    """``suffix(X, k)`` – dimensions from index *k* onwards."""
    shape, k = args
    shape = _to_shape(shape)
    return shape[k:]


def _broadcast_two(s1: list[int | None], s2: list[int | None]) -> list[int | None]:
    """NumPy-style shape broadcast."""
    result: list[int | None] = []
    r = max(len(s1), len(s2))
    s1 = [1] * (r - len(s1)) + list(s1)
    s2 = [1] * (r - len(s2)) + list(s2)
    for a, b in zip(s1, s2):
        if a is None or b is None:
            result.append(a if b == 1 else b if a == 1 else None)
        elif a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError(f"Broadcast incompatible: {a} vs {b}")
    return result


def _builtin_broadcast(args: list[Any]) -> list[int | None]:
    """``broadcast(s1, s2)``."""
    s1, s2 = args
    return _broadcast_two(_to_shape(s1), _to_shape(s2))


def _builtin_concat(args: list[Any]) -> list[int | None]:
    """``concat(s1, s2)`` – concatenate two shapes."""
    s1, s2 = args
    return _to_shape(s1) + _to_shape(s2)


def _builtin_permute(args: list[Any]) -> list[int | None]:
    """``permute(shape, perm)`` – reorder dimensions."""
    shape, perm = args
    shape = _to_shape(shape)
    perm = list(perm)
    return [shape[i] for i in perm]


def _builtin_normalize_axis(args: list[Any]) -> int:
    """``normalize_axis(axis, rank)`` – handle negative axis values."""
    axis, rank = args
    if axis < 0:
        axis += rank
    return axis


def _builtin_resolve_reshape(args: list[Any]) -> list[int | None]:
    """``resolve_reshape(input_shape, target)`` – ONNX reshape semantics.

    * ``0`` in *target* → copy from *input_shape* (unless *allowzero* is set).
    * ``-1`` in *target* → infer from total size.

    An optional third argument ``allowzero`` (default 0) controls whether
    ``0`` means "copy" (0) or literal zero (1).
    """
    input_shape = _to_shape(args[0])
    target = list(args[1])
    allowzero = args[2] if len(args) > 2 else 0

    # Handle 0s: copy dimension from input (unless allowzero)
    result = []
    for i, t in enumerate(target):
        if t == 0 and not allowzero and i < len(input_shape):
            result.append(input_shape[i])
        else:
            result.append(t)

    # Handle -1: infer from total
    if -1 in result:
        known_total = 1
        for d in input_shape:
            if d is None:
                known_total = None
                break
            known_total *= d

        if known_total is not None:
            known_output = 1
            neg_idx = -1
            for i, d in enumerate(result):
                if d == -1:
                    neg_idx = i
                elif d is None:
                    known_output = None
                    break
                else:
                    known_output *= d

            if known_output is not None and known_output != 0 and neg_idx >= 0:
                result[neg_idx] = known_total // known_output
            elif neg_idx >= 0:
                result[neg_idx] = None

    return result


def _builtin_squeeze_shape(args: list[Any]) -> list[int | None]:
    """``squeeze_shape(shape, axes)`` – remove dimensions at *axes*."""
    shape, axes = args
    shape = _to_shape(shape)
    axes = list(axes)
    rank = len(shape)
    normalised = {(a + rank if a < 0 else a) for a in axes}
    return [d for i, d in enumerate(shape) if i not in normalised]


def _builtin_unsqueeze_shape(args: list[Any]) -> list[int | None]:
    """``unsqueeze_shape(shape, axes)`` – insert 1-dimensions at *axes*."""
    shape, axes = args
    shape = _to_shape(shape)
    axes = list(axes)
    out_rank = len(shape) + len(axes)
    normalised = sorted((a + out_rank if a < 0 else a) for a in axes)
    result = list(shape)
    for offset, ax in enumerate(normalised):
        result.insert(ax, 1)
    return result


def _builtin_prod(args: list[Any]) -> int | None:
    """``prod(shape)`` – product of dimension values."""
    (vals,) = args
    vals = _to_shape(vals)
    result = 1
    for v in vals:
        if v is None:
            return None
        result *= v
    return result


def _builtin_unknown_nonneg(args: list[Any]) -> None:
    """``unknown_nonnegative()`` – unknown data-dependent dimension."""
    return _UNKNOWN


_BUILTINS: dict[str, Any] = {
    "shape": _builtin_shape,
    "dim": _builtin_dim,
    "rank": _builtin_rank,
    "prefix": _builtin_prefix,
    "suffix": _builtin_suffix,
    "broadcast": _builtin_broadcast,
    "concat": _builtin_concat,
    "permute": _builtin_permute,
    "normalize_axis": _builtin_normalize_axis,
    "shape_value": lambda args: list(args[0]),  # identity; value already resolved
    "resolve_reshape": _builtin_resolve_reshape,
    "squeeze_shape": _builtin_squeeze_shape,
    "unsqueeze_shape": _builtin_unsqueeze_shape,
    "prod": _builtin_prod,
    "sum": lambda args: sum(args[0]),
    "unknown_nonnegative": _builtin_unknown_nonneg,
    "range": lambda args: range(args[0]),
}


# ---------------------------------------------------------------------------
# AST expression evaluator
# ---------------------------------------------------------------------------


class _EvalEnv:
    """Evaluation environment for OSCL expressions."""

    def __init__(
        self,
        shapes: dict[str, list[int | None]],
        attributes: dict[str, Any],
        tensor_values: dict[str, list[int]] | None = None,
    ) -> None:
        self.shapes = dict(shapes)
        self.attributes = dict(attributes)
        self.tensor_values = dict(tensor_values or {})
        self.variables: dict[str, Any] = {}
        # Pre-populate variables with shapes (so identifiers resolve to shapes)
        self.variables.update(self.shapes)
        # Attributes are also accessible as variables
        self.variables.update(self.attributes)

    def lookup(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name]
        raise NameError(f"Undefined variable: {name!r}")


def _eval_expr(expr: Expr, env: _EvalEnv) -> Any:
    """Evaluate an OSCL expression tree in the given environment."""

    if isinstance(expr, NumberLit):
        return expr.value

    if isinstance(expr, UnknownDim):
        return _UNKNOWN

    if isinstance(expr, Identifier):
        return env.lookup(expr.name)

    if isinstance(expr, StringLit):
        return expr.value

    if isinstance(expr, ShapeLiteral):
        return [_eval_expr(d, env) for d in expr.dims]

    if isinstance(expr, IndexExpr):
        obj = _eval_expr(expr.obj, env)
        idx = _eval_expr(expr.index, env)
        return obj[idx]

    if isinstance(expr, BinOp):
        left = _eval_expr(expr.left, env)
        right = _eval_expr(expr.right, env)
        return _eval_binop(expr.op, left, right)

    if isinstance(expr, FuncCall):
        return _eval_func(expr, env)

    if isinstance(expr, IfExpr):
        cond = _eval_expr(expr.condition, env)
        if cond:
            return _eval_expr(expr.then_expr, env)
        return _eval_expr(expr.else_expr, env)

    if isinstance(expr, MapExpr):
        return _eval_map(expr, env)

    raise TypeError(f"Unknown expression type: {type(expr).__name__}")


def _eval_binop(op: str, left: Any, right: Any) -> Any:
    """Evaluate a binary operation."""
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "floordiv": lambda a, b: a // b,
        "ceildiv": lambda a, b: math.ceil(a / b),
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "and": lambda a, b: a and b,
        "or": lambda a, b: a or b,
        "max": lambda a, b: max(a, b),
        "min": lambda a, b: min(a, b),
    }
    if op not in ops:
        raise ValueError(f"Unsupported binary operator: {op!r}")
    return ops[op](left, right)


def _eval_func(call: FuncCall, env: _EvalEnv) -> Any:
    """Evaluate a function call."""
    name = call.name

    # shape_value needs special handling: resolve tensor values, not shape
    if name == "shape_value":
        arg_name = call.args[0]
        if isinstance(arg_name, Identifier) and arg_name.name in env.tensor_values:
            return list(env.tensor_values[arg_name.name])
        # Fall back to evaluating normally
        args = [_eval_expr(a, env) for a in call.args]
        return _BUILTINS[name](args)

    # resolve_reshape: inject allowzero attribute from environment
    if name == "resolve_reshape":
        args = [_eval_expr(a, env) for a in call.args]
        allowzero = env.attributes.get("allowzero", 0)
        args.append(allowzero)
        return _BUILTINS[name](args)

    # Regular built-in dispatch
    if name in _BUILTINS:
        args = [_eval_expr(a, env) for a in call.args]
        return _BUILTINS[name](args)

    raise NameError(f"Unknown function: {name!r}")


def _eval_map(expr: MapExpr, env: _EvalEnv) -> list[Any]:
    """Evaluate a map comprehension."""
    iterable = _eval_expr(expr.iter_expr, env)

    # range() returns a range object or int
    if isinstance(iterable, int):
        iterable = range(iterable)

    results = []
    for item in iterable:
        # Temporarily bind the loop variable
        old = env.variables.get(expr.var)
        env.variables[expr.var] = item
        results.append(_eval_expr(expr.body, env))
        # Restore
        if old is not None:
            env.variables[expr.var] = old
        else:
            env.variables.pop(expr.var, None)

    return results


# ---------------------------------------------------------------------------
# Spec execution
# ---------------------------------------------------------------------------


class ConstraintViolation(Exception):
    """Raised when a ``require`` constraint fails."""


def _execute_spec(
    spec: ShapeSpec,
    shapes: dict[str, list[int | None]],
    attributes: dict[str, Any],
    tensor_values: dict[str, list[int]] | None = None,
) -> dict[str, list[int | None]]:
    """Execute an OSCL spec and return the output shapes.

    Parameters
    ----------
    spec:
        Parsed OSCL shape specification.
    shapes:
        Mapping from input name (as declared in the spec) to its concrete shape.
    attributes:
        Mapping from attribute name to its value.
    tensor_values:
        Optional mapping from input name to its concrete integer values
        (needed for shape-tensor inputs like Reshape's ``shape`` input).

    Returns
    -------
    dict
        Mapping from output name to the inferred shape.
    """
    env = _EvalEnv(shapes, attributes, tensor_values)
    results: dict[str, list[int | None]] = {}

    for stmt in spec.statements:
        if isinstance(stmt, LetStmt):
            env.variables[stmt.name] = _eval_expr(stmt.expr, env)

        elif isinstance(stmt, RequireStmt):
            val = _eval_expr(stmt.expr, env)
            if not val:
                raise ConstraintViolation(f"Constraint violated: {stmt.expr}")

        elif isinstance(stmt, ResultStmt):
            val = _eval_expr(stmt.expr, env)
            results[stmt.name] = _to_shape(val)

        elif isinstance(stmt, WhenStmt):
            cond = _eval_expr(stmt.condition, env)
            if cond:
                for inner in stmt.body:
                    if isinstance(inner, LetStmt):
                        env.variables[inner.name] = _eval_expr(inner.expr, env)
                    elif isinstance(inner, ResultStmt):
                        val = _eval_expr(inner.expr, env)
                        results[inner.name] = _to_shape(val)
                    elif isinstance(inner, RequireStmt):
                        if not _eval_expr(inner.expr, env):
                            raise ConstraintViolation(
                                f"Constraint violated: {inner.expr}"
                            )

    return results


# ---------------------------------------------------------------------------
# ONNX helper utilities
# ---------------------------------------------------------------------------

# Mapping from ONNX op_type to OSCL spec name (lowercase).
_OP_TO_SPEC: dict[str, str] = {
    "Add": "add",
    "Concat": "concat",
    "Flatten": "flatten",
    "Gather": "gather",
    "Gemm": "gemm",
    "MatMul": "matmul",
    "NonZero": "nonzero",
    "Relu": "relu",
    "Reshape": "reshape",
    "Softmax": "softmax",
    "Squeeze": "squeeze",
    "Transpose": "transpose",
    "Unsqueeze": "unsqueeze",
}

# Default attribute values for operators (used when the attribute is absent).
_DEFAULT_ATTRS: dict[str, dict[str, Any]] = {
    "Flatten": {"axis": 1},
    "Gather": {"axis": 0},
    "Gemm": {"transA": 0, "transB": 0},
    "Softmax": {"axis": -1},
}


def _get_shape_from_type(tp: TypeProto) -> list[int | None] | None:
    """Extract a concrete shape from an ONNX TypeProto (None = unknown dim)."""
    if not tp.HasField("tensor_type"):
        return None
    tt = tp.tensor_type
    if not tt.HasField("shape"):
        return None
    dims: list[int | None] = []
    for d in tt.shape.dim:
        if d.dim_value > 0:
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(None)  # symbolic → unknown
        else:
            # dim_value == 0 can mean unknown or scalar dim
            dims.append(None)
    return dims


def _get_elem_type(tp: TypeProto) -> int:
    """Extract element type from ONNX TypeProto."""
    if tp.HasField("tensor_type"):
        return tp.tensor_type.elem_type
    return TensorProto.UNDEFINED


def _make_type_proto(
    shape: list[int | None], elem_type: int = TensorProto.FLOAT
) -> TypeProto:
    """Create an ONNX TypeProto from a shape list."""
    tp = TypeProto()
    tensor_tp = tp.tensor_type
    tensor_tp.elem_type = elem_type
    for d in shape:
        dim = tensor_tp.shape.dim.add()
        if d is not None:
            dim.dim_value = d
    return tp


def _get_initializer_values(initializer: TensorProto) -> list[int]:
    """Extract integer values from an ONNX initializer."""
    arr = numpy_helper.to_array(initializer)
    return arr.flatten().astype(int).tolist()


def _get_attribute_value(attr: onnx.AttributeProto) -> Any:
    """Extract a Python value from an ONNX AttributeProto."""
    if attr.type == onnx.AttributeProto.INT:
        return attr.i
    if attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    if attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OsclShapeInferenceEngine:
    """Shape inference engine backed by OSCL operator specifications.

    Usage::

        engine = OsclShapeInferenceEngine()
        inferred_model = engine.infer_shapes(model)
    """

    def __init__(self) -> None:
        self._specs = load_all_specs()

    @property
    def supported_ops(self) -> set[str]:
        """Set of ONNX op types supported by loaded OSCL specs."""
        return set(_OP_TO_SPEC.keys())

    def infer_shapes(self, model: ModelProto) -> ModelProto:
        """Infer shapes for all nodes in *model*.

        The interface mirrors ``onnx.shape_inference.infer_shapes``:
        returns a *new* ``ModelProto`` with inferred shapes populated in
        ``graph.value_info`` and ``graph.output``.
        """
        model = copy.deepcopy(model)
        graph = model.graph

        # Collect known shapes -------------------------------------------------
        known_shapes: dict[str, list[int | None]] = {}
        known_elem_types: dict[str, int] = {}

        for inp in graph.input:
            s = _get_shape_from_type(inp.type)
            if s is not None:
                known_shapes[inp.name] = s
            known_elem_types[inp.name] = _get_elem_type(inp.type)

        # Values from existing value_info
        for vi in graph.value_info:
            s = _get_shape_from_type(vi.type)
            if s is not None:
                known_shapes[vi.name] = s
            known_elem_types[vi.name] = _get_elem_type(vi.type)

        # Initialiser shapes and values ----------------------------------------
        initializer_values: dict[str, list[int]] = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            known_shapes[init.name] = list(arr.shape)
            known_elem_types.setdefault(init.name, init.data_type)
            # Keep integer values for shape-tensor resolution
            if init.data_type in (
                TensorProto.INT32,
                TensorProto.INT64,
                TensorProto.INT16,
                TensorProto.INT8,
            ):
                initializer_values[init.name] = arr.flatten().astype(int).tolist()

        # Forward pass: infer shapes node by node -----------------------------
        value_info_names = {vi.name for vi in graph.value_info}

        for node in graph.node:
            spec_name = _OP_TO_SPEC.get(node.op_type)
            if spec_name is None or spec_name not in self._specs:
                continue

            spec = self._specs[spec_name]

            # Map ONNX node inputs → OSCL spec input names
            input_shapes: dict[str, list[int | None]] = {}
            tensor_vals: dict[str, list[int]] = {}

            if spec.inputs and spec.inputs[0].variadic:
                # Variadic: all node inputs map to the variadic name
                var_name = spec.inputs[0].name
                var_shapes = []
                for onnx_in in node.input:
                    if onnx_in and onnx_in in known_shapes:
                        var_shapes.append(known_shapes[onnx_in])
                    else:
                        var_shapes.append([])
                input_shapes[var_name] = var_shapes  # type: ignore[assignment]
            else:
                for i, inp_decl in enumerate(spec.inputs):
                    if i < len(node.input) and node.input[i]:
                        onnx_name = node.input[i]
                        if onnx_name in known_shapes:
                            input_shapes[inp_decl.name] = known_shapes[onnx_name]
                        if onnx_name in initializer_values:
                            tensor_vals[inp_decl.name] = initializer_values[onnx_name]

            # Map node attributes → OSCL attribute names
            attrs: dict[str, Any] = {}
            # Start with default attribute values
            if node.op_type in _DEFAULT_ATTRS:
                attrs.update(_DEFAULT_ATTRS[node.op_type])
            node_attr_map = {a.name: a for a in node.attribute}
            for attr_name in spec.attributes:
                if attr_name in node_attr_map:
                    attrs[attr_name] = _get_attribute_value(node_attr_map[attr_name])

            # Special case: Transpose default perm = reverse order
            if node.op_type == "Transpose" and "perm" not in attrs:
                first_input = node.input[0] if node.input else ""
                if first_input in known_shapes:
                    rank = len(known_shapes[first_input])
                    attrs["perm"] = list(range(rank - 1, -1, -1))

            # Special case: Reshape allowzero attribute (not in OSCL spec but
            # needed for correct semantics).
            if node.op_type == "Reshape":
                attrs.setdefault("allowzero", 0)
                if "allowzero" in node_attr_map:
                    attrs["allowzero"] = _get_attribute_value(
                        node_attr_map["allowzero"]
                    )

            # Execute the spec --------------------------------------------------
            try:
                output_shapes = _execute_spec(
                    spec, input_shapes, attrs, tensor_vals
                )
            except Exception:
                continue  # graceful degradation

            # Store results back ------------------------------------------------
            for j, out_name in enumerate(spec.outputs):
                if j < len(node.output) and node.output[j]:
                    onnx_out = node.output[j]
                    inferred = output_shapes.get(out_name)
                    if inferred is not None:
                        known_shapes[onnx_out] = inferred

                        # Determine element type (inherit from first input)
                        et = TensorProto.UNDEFINED
                        if node.input:
                            et = known_elem_types.get(
                                node.input[0], TensorProto.UNDEFINED
                            )
                        known_elem_types[onnx_out] = et

                        # Update value_info or output
                        tp = _make_type_proto(inferred, et)
                        if onnx_out not in value_info_names:
                            vi = graph.value_info.add()
                            vi.name = onnx_out
                            vi.type.CopyFrom(tp)
                            value_info_names.add(onnx_out)

        # Patch graph outputs with inferred shapes ----------------------------
        for out in graph.output:
            if out.name in known_shapes:
                inferred = known_shapes[out.name]
                et = known_elem_types.get(out.name, _get_elem_type(out.type))
                out.type.CopyFrom(_make_type_proto(inferred, et))

        return model


# Module-level convenience function matching ``onnx.shape_inference.infer_shapes``.
_DEFAULT_ENGINE = None


def infer_shapes(model: ModelProto) -> ModelProto:
    """Infer shapes for *model* using OSCL specifications.

    Drop-in replacement for ``onnx.shape_inference.infer_shapes``.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = OsclShapeInferenceEngine()
    return _DEFAULT_ENGINE.infer_shapes(model)
