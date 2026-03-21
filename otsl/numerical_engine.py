"""OTSL-based numerical shape inference engine for ONNX models.

This module provides an :func:`infer_shapes` function whose interface mirrors
``onnx.shape_inference.infer_shapes``.  Internally it evaluates the bundled
OTSL operator specifications to compute output shapes numerically.
"""

from __future__ import annotations

import copy
import math
import operator
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
    NumberLit,
    RequireStmt,
    ResultStmt,
    ShapeLiteral,
    ShapeSpec,
    StringLit,
    UnknownDim,
)
from .loader import load_all_specs

__all__ = ["infer_shapes", "OtslNumericalShapeInferenceEngine"]

# Sentinel for unknown dimensions (data-dependent).
_UNKNOWN = None
DimValue = int | str | None


# ---------------------------------------------------------------------------
# Built-in OTSL functions
# ---------------------------------------------------------------------------


def _is_known_int(value: Any) -> bool:
    """Return ``True`` when *value* is a concrete integer dimension."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _to_shape(val: Any) -> list[DimValue]:
    """Coerce *val* to a shape (list of dimension ints or None)."""
    if isinstance(val, (list, tuple)):
        return list(val)
    raise TypeError(f"Cannot coerce {type(val).__name__} to shape: {val!r}")


def _builtin_shape(args: list[Any]) -> list[DimValue]:
    """``shape(X)`` - return the shape of tensor *X*."""
    (val,) = args
    return _to_shape(val)


def _builtin_dim(args: list[Any]) -> DimValue:
    """``dim(X, i)`` - return dimension *i* (supports negative indexing)."""
    shape, idx = args
    shape = _to_shape(shape)
    return shape[idx]


def _builtin_rank(args: list[Any]) -> int:
    """``rank(X)`` - number of dimensions."""
    (val,) = args
    return len(_to_shape(val))


def _builtin_prefix(args: list[Any]) -> list[DimValue]:
    """``prefix(X, k)`` - first *k* dimensions (negative *k* -> all but last |k|)."""
    shape, k = args
    shape = _to_shape(shape)
    if k < 0:
        end = len(shape) + k
        return shape[:max(end, 0)]
    return shape[:k]


def _builtin_suffix(args: list[Any]) -> list[DimValue]:
    """``suffix(X, k)`` - dimensions from index *k* onwards."""
    shape, k = args
    shape = _to_shape(shape)
    return shape[k:]


def _broadcast_two(s1: list[DimValue], s2: list[DimValue]) -> list[DimValue]:
    """NumPy-style shape broadcast."""
    result: list[DimValue] = []
    r = max(len(s1), len(s2))
    s1 = [1] * (r - len(s1)) + list(s1)
    s2 = [1] * (r - len(s2)) + list(s2)
    for a, b in zip(s1, s2):
        if not _is_known_int(a) or not _is_known_int(b):
            if a == b:
                result.append(a)
            else:
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
    """``concat(s1, s2)`` - concatenate two shapes."""
    s1, s2 = args
    return _to_shape(s1) + _to_shape(s2)


def _builtin_concat_shape(args: list[Any]) -> list[int | None]:
    """Compute ONNX Concat output shape from a variadic input family and axis."""
    inputs, axis = args
    if not inputs:
        raise ValueError("concat_shape requires at least one input")
    first = _to_shape(inputs[0])
    r = len(first)
    if axis < 0:
        axis += r
    result: list[int | None] = []
    for j in range(r):
        if j == axis:
            total: int | None = 0
            for inp in inputs:
                d = _to_shape(inp)[j]
                if d is None or total is None:
                    total = None
                else:
                    total += d
            result.append(total)
        else:
            result.append(first[j])
    return result


def _builtin_permute(args: list[Any]) -> list[int | None]:
    """``permute(shape, perm)`` - reorder dimensions."""
    shape, perm = args
    shape = _to_shape(shape)
    perm = list(perm)
    return [shape[i] for i in perm]


def _builtin_normalize_axis(args: list[Any]) -> int:
    """``normalize_axis(axis, rank)`` - handle negative axis values."""
    axis, rank = args
    if axis < 0:
        axis += rank
    return axis


def _builtin_reverse_perm(args: list[Any]) -> list[int]:
    """``reverse_perm(rank)`` - default transpose permutation."""
    (rank,) = args
    return list(range(rank - 1, -1, -1))


def _builtin_ones(args: list[Any]) -> list[int]:
    """``ones(rank)`` - list of ones of length *rank*."""
    (rank,) = args
    return [1] * rank


def _builtin_iota(args: list[Any]) -> list[int]:
    """``iota(n)`` - ascending integer list [0, ..., n-1]."""
    (count,) = args
    return list(range(int(count)))


def _builtin_repeat(args: list[Any]) -> list[Any]:
    """``repeat(value, count)`` - repeat a value count times."""
    value, count = args
    return [value] * int(count)


def _builtin_length(args: list[Any]) -> int:
    """``length(value)`` - length of a list-like value."""
    (value,) = args
    return len(value)


def _eval_output_count_func(env: _EvalEnv) -> int:
    """Evaluate ``output_count()`` for the current ONNX node."""
    return len(env.node_output_types)


def _builtin_overlay(args: list[Any]) -> list[int | None]:
    """``overlay(values, base, axes)`` - replace base axes with values."""
    values, base, axes = args
    base_shape = list(_to_shape(base))
    values = list(values)
    axes = list(axes)
    if not values:
        return base_shape
    if not axes:
        return list(_to_shape(values))
    for axis, value in zip(axes, values):
        ax = int(axis)
        if ax < 0:
            ax += len(base_shape)
        base_shape[ax] = value
    return base_shape


def _builtin_floordiv(args: list[Any]) -> int | None:
    """``floordiv(a, b)`` - integer division with unknown propagation."""
    left, right = args
    if left is None or right in (None, 0):
        return None
    return left // right


def _builtin_resolve_reshape(args: list[Any]) -> list[int | None]:
    """``resolve_reshape(input_shape, target)`` - ONNX reshape semantics.

    * ``0`` in *target* -> copy from *input_shape* (unless *allowzero* is set).
    * ``-1`` in *target* -> infer from total size.

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
    """``squeeze_shape(shape, axes)`` - remove dimensions at *axes*."""
    shape, axes = args
    shape = _to_shape(shape)
    axes = list(axes)
    rank = len(shape)
    normalised = {(a + rank if a < 0 else a) for a in axes}
    return [d for i, d in enumerate(shape) if i not in normalised]


def _builtin_unsqueeze_shape(args: list[Any]) -> list[int | None]:
    """``unsqueeze_shape(shape, axes)`` - insert 1-dimensions at *axes*."""
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
    """``prod(shape)`` - product of dimension values."""
    vals = _to_shape(args[0])
    return None if any(v is None for v in vals) else math.prod(vals)


def _builtin_subshape(args: list[Any]) -> list[int | None]:
    """``subshape(shape, start, end)`` - slice a shape list."""
    shape, start, end = args
    dims = _to_shape(shape)
    rank = len(dims)

    if start is None:
        start = 0
    if end is None:
        end = rank

    start = int(start)
    end = int(end)

    if start < 0:
        start = max(start + rank, 0)
    if end < 0:
        end = max(end + rank, 0)

    start = min(max(start, 0), rank)
    end = min(max(end, 0), rank)
    if end < start:
        end = start

    return dims[start:end]


def _builtin_unknown_nonnegative(args: list[Any]) -> None:
    """``unknown_nonnegative()`` - unknown data-dependent dimension."""
    return _UNKNOWN


def _builtin_reduce_shape(args: list[Any]) -> list[int | None]:
    """``reduce_shape(shape, axes, keepdims)`` - compute output shape of reduction."""
    shape, axes, keepdims = args
    shape = _to_shape(shape)
    rank = len(shape)
    if not axes:
        # Empty axes -> reduce all dimensions
        normalised = set(range(rank))
    else:
        axes = list(axes)
        normalised = {(a + rank if a < 0 else a) for a in axes}
    result: list[int | None] = []
    for i, d in enumerate(shape):
        if i in normalised:
            if keepdims:
                result.append(1)
            # else: skip (dimension removed)
        else:
            result.append(d)
    return result


def _builtin_tile_shape(args: list[Any]) -> list[int | None]:
    """``tile_shape(shape, repeats)`` - multiply dims by repeats."""
    shape, repeats = args
    shape = _to_shape(shape)
    repeats = list(repeats)
    # Pad to same length
    while len(shape) < len(repeats):
        shape = [1] + shape
    while len(repeats) < len(shape):
        repeats = [1] + repeats
    return [
        (d * r if d is not None and r is not None else None)
        for d, r in zip(shape, repeats)
    ]


def _builtin_slice_shape(args: list[Any]) -> list[int | None]:
    """``slice_shape(shape, starts, ends, axes, steps)`` - ONNX Slice output shape."""
    shape, starts, ends, axes, steps = args
    shape = _to_shape(shape)
    rank = len(shape)
    result = list(shape)
    for i in range(len(axes)):
        ax = axes[i]
        if ax < 0:
            ax += rank
        s = starts[i]
        e = ends[i]
        st = steps[i] if i < len(steps) else 1
        dim_size = shape[ax]
        if dim_size is None:
            result[ax] = None
            continue
        if st > 0:
            # Clamp for positive step
            if s < 0:
                s = max(s + dim_size, 0)
            else:
                s = min(s, dim_size)
            if e < 0:
                e = max(e + dim_size, 0)
            else:
                e = min(e, dim_size)
            result[ax] = max(0, math.ceil((e - s) / st))
        elif st < 0:
            # Clamp for negative step
            if s < 0:
                s = max(s + dim_size, -1)
            else:
                s = min(s, dim_size - 1)
            if e < 0:
                e = max(e + dim_size, -1)
            else:
                e = min(e, dim_size - 1)
            if s > e:
                result[ax] = max(0, math.ceil((s - e) / (-st)))
            else:
                result[ax] = 0
        else:
            result[ax] = None
    return result


def _builtin_pad_shape(args: list[Any]) -> list[int | None]:
    """``pad_shape(shape, pads, axes)`` - add padding to shape dimensions."""
    shape = _to_shape(args[0])
    pads = list(args[1])
    axes = list(args[2]) if len(args) > 2 and args[2] else None
    rank = len(shape)

    if axes is not None:
        n = len(axes)
        normalised = [(a + rank if a < 0 else a) for a in axes]
        result = list(shape)
        for j, ax in enumerate(normalised):
            if 0 <= ax < rank:
                d = shape[ax]
                begin = pads[j] if j < len(pads) else 0
                end = pads[j + n] if (j + n) < len(pads) else 0
                result[ax] = (d + begin + end) if d is not None else None
        return result

    # pads format: [x1_begin, x2_begin, ..., xn_begin, x1_end, ..., xn_end]
    result: list[int | None] = []
    for i in range(rank):
        d = shape[i]
        if d is None:
            result.append(None)
        else:
            begin = pads[i] if i < len(pads) else 0
            end = pads[i + rank] if (i + rank) < len(pads) else 0
            result.append(d + begin + end)
    return result


def _builtin_split_shape(args: list[Any]) -> list[int | None]:
    """``split_shape(shape, axis, sizes)`` - compute first split output shape.

    For the Split operator we only compute the *first* output here.
    The engine handles the multi-output case specially.
    """
    shape, axis, sizes = args
    shape = _to_shape(shape)
    result = list(shape)
    if sizes:
        result[axis] = sizes[0]
    return result


def _builtin_split_shapes(args: list[Any]) -> list[list[int | None]]:
    """``split_shapes(shape, axis, sizes, num_outputs)`` - compute all split shapes."""
    input_shape, axis, sizes, num_outputs = args
    shape = _to_shape(input_shape)
    axis = int(axis)
    rank = len(shape)
    if axis < 0:
        axis += rank

    split_sizes = list(sizes) if sizes else []
    if not split_sizes:
        count = int(num_outputs)
        if count <= 0:
            return []
        dim = shape[axis]
        if dim is None:
            split_sizes = [None] * count
        else:
            base = dim // count
            remainder = dim % count
            split_sizes = [base + (1 if i < remainder else 0) for i in range(count)]

    outputs: list[list[int | None]] = []
    for size in split_sizes:
        out_shape = list(shape)
        out_shape[axis] = size
        outputs.append(out_shape)
    return outputs


def _builtin_pool_shape(args: list[Any]) -> list[int | None]:
    """``pool_shape(input, kernel, strides, pads, dilations, ceil_mode, auto_pad)``."""
    input_shape, kernel_shape, strides, pads, dilations, ceil_mode, auto_pad = args
    input_shape = _to_shape(input_shape)
    rank = len(input_shape)
    spatial_rank = rank - 2  # N, C, spatial...

    if not strides:
        strides = [1] * spatial_rank
    if not pads:
        pads = [0] * (2 * spatial_rank)
    if not dilations:
        dilations = [1] * spatial_rank

    # Batch and channels stay the same
    result = [input_shape[0], input_shape[1]]

    for i in range(spatial_rank):
        d = input_shape[i + 2]
        if d is None:
            result.append(None)
            continue
        k = kernel_shape[i] if i < len(kernel_shape) else 1
        s = strides[i] if i < len(strides) else 1
        dil = dilations[i] if i < len(dilations) else 1
        pad_begin = pads[i] if i < len(pads) else 0
        pad_end = pads[i + spatial_rank] if (i + spatial_rank) < len(pads) else 0

        effective_k = (k - 1) * dil + 1

        if isinstance(auto_pad, str) and auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out = math.ceil(d / s)
        elif isinstance(auto_pad, str) and auto_pad == "VALID":
            out = math.ceil((d - effective_k + 1) / s)
        else:
            # NOTSET or explicit pads
            if ceil_mode:
                out = math.ceil((d + pad_begin + pad_end - effective_k) / s) + 1
                # Adjust if the last pooling starts inside padding
                if (out - 1) * s >= d + pad_begin:
                    out -= 1
            else:
                out = math.floor((d + pad_begin + pad_end - effective_k) / s) + 1
        result.append(max(out, 0))
    return result


def _builtin_global_pool_shape(args: list[Any]) -> list[int | None]:
    """``global_pool_shape(input)`` - [N, C, 1, 1, ...]."""
    (input_shape,) = args
    input_shape = _to_shape(input_shape)
    spatial_rank = len(input_shape) - 2
    return input_shape[:2] + [1] * spatial_rank


def _builtin_conv_shape(args: list[Any]) -> list[int | None]:
    """``conv_shape(X, W, kernel_shape, strides, pads, dilations, group, auto_pad)``."""
    x_shape, w_shape, kernel_shape, strides, pads, dilations, group, auto_pad = args
    x_shape = _to_shape(x_shape)
    w_shape = _to_shape(w_shape)
    spatial_rank = len(x_shape) - 2

    if not kernel_shape:
        kernel_shape = w_shape[2:]  # infer from weight
    if not strides:
        strides = [1] * spatial_rank
    if not pads:
        pads = [0] * (2 * spatial_rank)
    if not dilations:
        dilations = [1] * spatial_rank

    # Output: [N, C_out, spatial...]
    result = [x_shape[0], w_shape[0]]  # N, num_filters

    for i in range(spatial_rank):
        d = x_shape[i + 2]
        if d is None:
            result.append(None)
            continue
        k = kernel_shape[i] if i < len(kernel_shape) else 1
        s = strides[i] if i < len(strides) else 1
        dil = dilations[i] if i < len(dilations) else 1
        pad_begin = pads[i] if i < len(pads) else 0
        pad_end = pads[i + spatial_rank] if (i + spatial_rank) < len(pads) else 0

        effective_k = (k - 1) * dil + 1

        if isinstance(auto_pad, str) and auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out = math.ceil(d / s)
        elif isinstance(auto_pad, str) and auto_pad == "VALID":
            out = math.ceil((d - effective_k + 1) / s)
        else:
            out = math.floor((d + pad_begin + pad_end - effective_k) / s) + 1
        result.append(max(out, 0))
    return result


def _builtin_convtranspose_shape(args: list[Any]) -> list[int | None]:
    """``convtranspose_shape(...)``."""
    (x_shape, w_shape, kernel_shape, strides, pads, dilations,
     group, auto_pad, output_padding, output_shape_attr) = args
    x_shape = _to_shape(x_shape)
    w_shape = _to_shape(w_shape)
    spatial_rank = len(x_shape) - 2

    if not kernel_shape:
        kernel_shape = w_shape[2:]
    if not strides:
        strides = [1] * spatial_rank
    if not pads:
        pads = [0] * (2 * spatial_rank)
    if not dilations:
        dilations = [1] * spatial_rank
    if not output_padding:
        output_padding = [0] * spatial_rank

    result = [x_shape[0], w_shape[1] * group]  # N, C_out

    if output_shape_attr:
        # output_shape attribute overrides calculation
        for i in range(spatial_rank):
            result.append(output_shape_attr[i])
        return result

    for i in range(spatial_rank):
        d = x_shape[i + 2]
        if d is None:
            result.append(None)
            continue
        k = kernel_shape[i] if i < len(kernel_shape) else 1
        s = strides[i] if i < len(strides) else 1
        dil = dilations[i] if i < len(dilations) else 1
        pad_begin = pads[i] if i < len(pads) else 0
        pad_end = pads[i + spatial_rank] if (i + spatial_rank) < len(pads) else 0
        opad = output_padding[i] if i < len(output_padding) else 0

        effective_k = (k - 1) * dil + 1

        if isinstance(auto_pad, str) and auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            out = d * s
        elif isinstance(auto_pad, str) and auto_pad == "VALID":
            out = d * s + max(effective_k - s, 0)
        else:
            out = (d - 1) * s + effective_k - pad_begin - pad_end + opad
        result.append(max(out, 0))
    return result


def _builtin_depthtospace_shape(args: list[Any]) -> list[int | None]:
    """``depthtospace_shape(shape, blocksize)``."""
    shape, bs = args
    shape = _to_shape(shape)
    # [N, C, H, W] -> [N, C/(bs*bs), H*bs, W*bs]
    if len(shape) < 4:
        return shape
    n, c = shape[0], shape[1]
    result = [n, c // (bs * bs) if c is not None else None]
    for d in shape[2:]:
        result.append(d * bs if d is not None else None)
    return result


def _builtin_spacetodepth_shape(args: list[Any]) -> list[int | None]:
    """``spacetodepth_shape(shape, blocksize)``."""
    shape, bs = args
    shape = _to_shape(shape)
    if len(shape) < 4:
        return shape
    n, c = shape[0], shape[1]
    result = [n, c * bs * bs if c is not None else None]
    for d in shape[2:]:
        result.append(d // bs if d is not None else None)
    return result


def _builtin_topk_shape(args: list[Any]) -> list[int | None]:
    """``topk_shape(shape, axis, k_value)``."""
    shape, axis, k_value = args
    shape = _to_shape(shape)
    rank = len(shape)
    if axis < 0:
        axis += rank
    result = list(shape)
    if isinstance(k_value, list):
        k_value = k_value[0] if k_value else None
    result[axis] = k_value
    return result


def _builtin_onehot_shape(args: list[Any]) -> list[int | None]:
    """``onehot_shape(indices_shape, depth, axis)``."""
    indices_shape, depth, axis = args
    indices_shape = _to_shape(indices_shape)
    if isinstance(depth, list):
        depth = depth[0] if depth else None
    if isinstance(depth, float) and depth.is_integer():
        depth = int(depth)
    result = list(indices_shape)
    if axis < 0:
        axis += len(result) + 1
    result.insert(axis, depth)
    return result


def _builtin_gathernd_shape(args: list[Any]) -> list[int | None]:
    """``gathernd_shape(data_shape, indices_shape, batch_dims)``."""
    data_shape, indices_shape, batch_dims = args
    data_shape = _to_shape(data_shape)
    indices_shape = _to_shape(indices_shape)
    # Last dim of indices
    last_idx_dim = indices_shape[-1] if indices_shape else 0
    if last_idx_dim is None:
        return [None]
    # batch dims
    batch_shape = indices_shape[:batch_dims] if batch_dims else []
    # Result shape: batch_dims + indices[batch_dims:-1] + data[batch_dims + last_idx_dim:]
    result = list(batch_shape) + indices_shape[batch_dims:-1] + data_shape[batch_dims + last_idx_dim:]
    return result


def _builtin_nll_loss_shape(args: list[Any]) -> list[int | None]:
    """``nll_loss_shape(input_shape, target_shape, reduction)``."""
    input_shape, target_shape, reduction = args
    input_shape = _to_shape(input_shape)
    if reduction in ("mean", "sum"):
        return []  # scalar
    # "none": output shape = target_shape
    return _to_shape(target_shape)


def _builtin_resize_shape(args: list[Any]) -> list[int | None]:
    """``resize_shape(input_shape, scales, sizes)``."""
    input_shape, scales, sizes = args
    input_shape = _to_shape(input_shape)

    if isinstance(sizes, list) and sizes:
        return list(sizes)

    if isinstance(scales, list) and scales:
        result = []
        for i, d in enumerate(input_shape):
            if d is None or i >= len(scales):
                result.append(None)
            else:
                s = scales[i]
                result.append(int(math.floor(d * s)))
        return result

    return input_shape


def _builtin_rnn_shape(args: list[Any]) -> list[int | None]:
    """``rnn_shape(X_shape, hidden_size, num_directions)``."""
    x_shape, hidden_size, num_directions = args
    x_shape = _to_shape(x_shape)
    # X: [seq_length, batch_size, input_size]
    seq_length = x_shape[0] if len(x_shape) > 0 else None
    batch_size = x_shape[1] if len(x_shape) > 1 else None
    # Y: [seq_length, num_directions, batch_size, hidden_size]
    return [seq_length, num_directions, batch_size, hidden_size]


def _builtin_roialign_shape(args: list[Any]) -> list[int | None]:
    """``roialign_shape(X_shape, rois_shape, output_height, output_width)``."""
    x_shape, rois_shape, out_h, out_w = args
    x_shape = _to_shape(x_shape)
    rois_shape = _to_shape(rois_shape)
    num_rois = rois_shape[0] if rois_shape else None
    channels = x_shape[1] if len(x_shape) > 1 else None
    return [num_rois, channels, out_h, out_w]


def _builtin_gridsample_shape(args: list[Any]) -> list[int | None]:
    """``gridsample_shape(X_shape, grid_shape)``."""
    x_shape, grid_shape = args
    x_shape = _to_shape(x_shape)
    grid_shape = _to_shape(grid_shape)
    # X: [N, C, ...], grid: [N, D1, D2, ..., ndim]
    result = [x_shape[0], x_shape[1]]  # N, C
    for d in grid_shape[1:-1]:
        result.append(d)
    return result


def _builtin_einsum_shape(args: list[Any]) -> list[int | None]:
    """``einsum_shape(Xs, equation)`` - compute output shape from einsum equation."""
    xs, equation = args
    if isinstance(equation, bytes):
        equation = equation.decode("utf-8")
    # Parse equation: "ij,jk->ik"
    if "->" not in equation:
        return [_UNKNOWN]
    lhs, rhs = equation.split("->")
    input_specs = lhs.split(",")

    # Build dimension mapping
    dim_map: dict[str, int | None] = {}
    for spec_idx, spec in enumerate(input_specs):
        spec = spec.strip()
        if spec_idx < len(xs):
            shape = _to_shape(xs[spec_idx])
        else:
            continue
        ellipsis_pos = spec.find("...")
        if ellipsis_pos >= 0:
            # Handle ellipsis - skip for now
            prefix_chars = spec[:ellipsis_pos]
            suffix_chars = spec[ellipsis_pos + 3:]
            for ci, c in enumerate(prefix_chars):
                if ci < len(shape):
                    dim_map[c] = shape[ci]
            for ci, c in enumerate(reversed(suffix_chars)):
                if ci < len(shape):
                    dim_map[c] = shape[len(shape) - 1 - ci]
        else:
            for ci, c in enumerate(spec):
                if ci < len(shape):
                    dim_map[c] = shape[ci]

    rhs = rhs.strip()
    result: list[int | None] = []
    i = 0
    while i < len(rhs):
        if rhs[i:i + 3] == "...":
            # Ellipsis in output - find broadcast dims
            i += 3
            continue
        c = rhs[i]
        if c in dim_map:
            result.append(dim_map[c])
        else:
            result.append(None)
        i += 1
    return result


def _builtin_affinegrid_shape(args: list[Any]) -> list[int | None]:
    """``affinegrid_shape(theta_shape, size)``."""
    theta_shape, size = args
    theta_shape = _to_shape(theta_shape)
    size = list(size)
    # theta: [N, spatial_dims, spatial_dims+1], size: [N, C, D1, D2, ...]
    n = theta_shape[0] if theta_shape else size[0] if size else None
    spatial = size[2:]  # spatial dimensions
    # grid: [N, D1, ..., Dn, ndim]
    result = [n] + spatial + [len(spatial)]
    return result


def _builtin_dft_shape(args: list[Any]) -> list[int | None]:
    """``dft_shape(input_shape, axis, onesided)``."""
    input_shape, axis, onesided = args
    input_shape = _to_shape(input_shape)
    result = list(input_shape)
    rank = len(result)
    if axis < 0:
        axis += rank
    # The last dim encodes real (1) or complex (2).
    # Output is always complex, so last dim becomes 2.
    if result and result[-1] == 1:
        result[-1] = 2
    elif not result or result[-1] != 2:
        result.append(2)
    if onesided and 0 <= axis < len(result):
        d = result[axis]
        if d is not None:
            result[axis] = d // 2 + 1
    return result


def _builtin_stft_shape(args: list[Any]) -> list[int | None]:
    """``stft_shape(signal_shape, frame_step, window_shape, onesided)``."""
    signal_shape, frame_step_vals, window_shape, onesided = args
    signal_shape = _to_shape(signal_shape)
    window_shape = _to_shape(window_shape) if window_shape else []
    batch = signal_shape[0] if signal_shape else None
    signal_length = signal_shape[1] if len(signal_shape) > 1 else None

    # Determine frame_length from window shape
    frame_length: int | None = window_shape[0] if window_shape else None

    # frame_step is a scalar tensor value passed via shape_value()
    step: int | None = None
    if isinstance(frame_step_vals, list) and len(frame_step_vals) == 1:
        step = int(float(frame_step_vals[0]))

    # num_frames = (signal_length - frame_length) / frame_step + 1
    num_frames: int | None = None
    if signal_length is not None and frame_length is not None and step:
        num_frames = max(0, (signal_length - frame_length) // step + 1)

    # fft_length depends on onesided flag
    fft_length: int | None = None
    if frame_length is not None:
        fft_length = frame_length // 2 + 1 if onesided else frame_length

    return [batch, num_frames, fft_length, 2]


def _builtin_col2im_shape(args: list[Any]) -> list[int | None]:
    """``col2im_shape(data_shape, image_shape, block_shape)``."""
    data_shape, image_shape, block_shape = args
    data_shape = _to_shape(data_shape)
    image_shape = list(image_shape)
    # data: [N, C*product(block_shape), L], output: [N, C, image_shape...]
    n = data_shape[0] if data_shape else None
    block_prod = 1
    for b in list(block_shape):
        block_prod *= b
    c_times_block = data_shape[1] if len(data_shape) > 1 else None
    c = c_times_block // block_prod if c_times_block is not None else None
    return [n, c] + image_shape


def _builtin_range_output_shape(args: list[Any]) -> list[int | None]:
    """``range_output_shape(start, limit, delta)``."""
    start_vals, limit_vals, delta_vals = args
    # start/limit/delta are scalar tensor values passed via shape_value().
    if (
        isinstance(start_vals, list) and len(start_vals) == 1
        and isinstance(limit_vals, list) and len(limit_vals) == 1
        and isinstance(delta_vals, list) and len(delta_vals) == 1
    ):
        start = start_vals[0]
        limit = limit_vals[0]
        delta = delta_vals[0]
        if delta != 0:
            return [max(0, math.ceil((limit - start) / delta))]
    return [_UNKNOWN]


def _builtin_sequence_type(args: list[Any]) -> TypeProto:
    """``sequence_type(shape, elem_type)`` - construct a sequence-of-tensor type."""
    shape, elem_type = args
    return _make_sequence_type_proto(_to_shape(shape), int(elem_type))


_BUILTINS: dict[str, Any] = {
    "shape": _builtin_shape,
    "dim": _builtin_dim,
    "rank": _builtin_rank,
    "prefix": _builtin_prefix,
    "suffix": _builtin_suffix,
    "broadcast": _builtin_broadcast,
    "concat": _builtin_concat,
    "concat_shape": _builtin_concat_shape,
    "permute": _builtin_permute,
    "normalize_axis": _builtin_normalize_axis,
    "reverse_perm": _builtin_reverse_perm,
    "ones": _builtin_ones,
    "iota": _builtin_iota,
    "repeat": _builtin_repeat,
    "length": _builtin_length,
    "overlay": _builtin_overlay,
    "floordiv": _builtin_floordiv,
    "shape_value": lambda args: list(args[0]),  # identity; value already resolved
    "resolve_reshape": _builtin_resolve_reshape,
    "squeeze_shape": _builtin_squeeze_shape,
    "unsqueeze_shape": _builtin_unsqueeze_shape,
    "prod": _builtin_prod,
    "subshape": _builtin_subshape,
    "unknown_nonnegative": _builtin_unknown_nonnegative,
    # --- new built-ins ---
    "reduce_shape": _builtin_reduce_shape,
    "tile_shape": _builtin_tile_shape,
    "slice_shape": _builtin_slice_shape,
    "pad_shape": _builtin_pad_shape,
    "split_shape": _builtin_split_shape,
    "split_shapes": _builtin_split_shapes,
    "pool_shape": _builtin_pool_shape,
    "global_pool_shape": _builtin_global_pool_shape,
    "conv_shape": _builtin_conv_shape,
    "convtranspose_shape": _builtin_convtranspose_shape,
    "depthtospace_shape": _builtin_depthtospace_shape,
    "spacetodepth_shape": _builtin_spacetodepth_shape,
    "topk_shape": _builtin_topk_shape,
    "onehot_shape": _builtin_onehot_shape,
    "gathernd_shape": _builtin_gathernd_shape,
    "nll_loss_shape": _builtin_nll_loss_shape,
    "resize_shape": _builtin_resize_shape,
    "rnn_shape": _builtin_rnn_shape,
    "roialign_shape": _builtin_roialign_shape,
    "gridsample_shape": _builtin_gridsample_shape,
    "einsum_shape": _builtin_einsum_shape,
    "affinegrid_shape": _builtin_affinegrid_shape,
    "dft_shape": _builtin_dft_shape,
    "stft_shape": _builtin_stft_shape,
    "col2im_shape": _builtin_col2im_shape,
    "range_output_shape": _builtin_range_output_shape,
    "sequence_type": _builtin_sequence_type,
}


# ---------------------------------------------------------------------------
# Type name constants  (bare identifiers such as ``int64`` or ``bool``)
# ---------------------------------------------------------------------------

_TYPE_NAME_CONSTANTS: dict[str, int] = {
    "undefined": TensorProto.UNDEFINED,
    "float": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "double": TensorProto.DOUBLE,
    "int64": TensorProto.INT64,
    "int32": TensorProto.INT32,
    "int16": TensorProto.INT16,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
    "uint16": TensorProto.UINT16,
    "uint32": TensorProto.UINT32,
    "uint64": TensorProto.UINT64,
    "bool": TensorProto.BOOL,
    "bfloat16": TensorProto.BFLOAT16,
    "complex64": TensorProto.COMPLEX64,
    "complex128": TensorProto.COMPLEX128,
    "string": TensorProto.STRING,
}


# ---------------------------------------------------------------------------
# AST expression evaluator
# ---------------------------------------------------------------------------


class _EvalEnv:
    """Evaluation environment for OTSL expressions."""

    def __init__(
        self,
        shapes: dict[str, list[int | None]],
        attributes: dict[str, Any],
        tensor_values: dict[str, list[int]] | None = None,
        elem_types: dict[str, Any] | None = None,
        attribute_protos: dict[str, Any] | None = None,
        full_types: dict[str, Any] | None = None,
        node_input_types: list[Any] | None = None,
        node_output_types: list[Any] | None = None,
    ) -> None:
        self.shapes = dict(shapes)
        self.attributes = dict(attributes)
        self.tensor_values = dict(tensor_values or {})
        self.elem_types: dict[str, Any] = dict(elem_types or {})
        self.attribute_protos: dict[str, Any] = dict(attribute_protos or {})
        self.full_types: dict[str, Any] = dict(full_types or {})
        self.node_input_types: list[Any] = list(node_input_types or [])
        self.node_output_types: list[Any] = list(node_output_types or [])
        self.variables: dict[str, Any] = {}
        # Type name constants (resolve bare identifiers like ``int64``)
        self.variables.update(_TYPE_NAME_CONSTANTS)
        # Pre-populate variables with shapes (so identifiers resolve to shapes)
        self.variables.update(self.shapes)
        # Attributes are also accessible as variables
        self.variables.update(self.attributes)
        self._fresh_unknown_index = 0

    def lookup(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name]
        raise NameError(f"Undefined variable: {name!r}")

    def fresh_unknown(self) -> str:
        name = f"unk__{self._fresh_unknown_index}"
        self._fresh_unknown_index += 1
        return name


def _eval_expr(expr: Expr, env: _EvalEnv) -> Any:
    """Evaluate an OTSL expression tree in the given environment."""

    if isinstance(expr, NumberLit):
        return expr.value

    if isinstance(expr, UnknownDim):
        return env.fresh_unknown()

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

    raise TypeError(f"Unknown expression type: {type(expr).__name__}")


def _eval_binop(op: str, left: Any, right: Any) -> Any:
    """Evaluate a binary operation."""
    fn = _BINOP_DISPATCH.get(op)
    if fn is None:
        raise ValueError(f"Unsupported binary operator: {op!r}")
    return fn(left, right)


# Hoisted dispatch table for binary operations (avoids per-call dict creation).
_BINOP_DISPATCH: dict[str, Any] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "floordiv": operator.floordiv,
    "ceildiv": lambda a, b: math.ceil(a / b),
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "max": max,
    "min": min,
}


def _eval_type_func(arg: Expr, env: _EvalEnv) -> int:
    """Evaluate ``type(expr)`` by looking up the element type of the tensor."""
    if isinstance(arg, Identifier):
        et = env.elem_types.get(arg.name)
        if et is not None:
            return et
    elif isinstance(arg, IndexExpr) and isinstance(arg.obj, Identifier):
        et = env.elem_types.get(arg.obj.name)
        if isinstance(et, list):
            idx = _eval_expr(arg.index, env)
            if 0 <= idx < len(et):
                return et[idx]
    return TensorProto.UNDEFINED


def _eval_full_type_func(arg: Expr, env: _EvalEnv) -> TypeProto | None:
    """Evaluate ``full_type(expr)`` by looking up the full ONNX type."""
    if isinstance(arg, Identifier):
        tp = env.full_types.get(arg.name)
        if tp is not None:
            return copy.deepcopy(tp)
    return None


def _eval_unwrap_optional_type_func(call: FuncCall, env: _EvalEnv) -> TypeProto | None:
    """Evaluate ``unwrap_optional_type(x)`` by stripping one optional layer."""
    tp = _eval_full_type_func(call.args[0], env)
    if tp is None:
        return None
    if tp.HasField("optional_type"):
        return copy.deepcopy(tp.optional_type.elem_type)
    return copy.deepcopy(tp)


def _unwrap_optional_type_proto(tp: TypeProto) -> TypeProto:
    """Strip one optional wrapper from a TypeProto if present."""
    if tp.HasField("optional_type"):
        return copy.deepcopy(tp.optional_type.elem_type)
    return copy.deepcopy(tp)


def _eval_if_output_types_func(env: _EvalEnv) -> list[TypeProto]:
    """Evaluate merged output types for an If node from branch graph attrs."""
    then_branch = env.attribute_protos.get("then_branch")
    else_branch = env.attribute_protos.get("else_branch")
    if then_branch is None or else_branch is None:
        return []
    result: list[TypeProto] = []
    for then_out, else_out in zip(then_branch.g.output, else_branch.g.output):
        result.append(_merge_type_protos(then_out.type, else_out.type))
    return result


def _eval_loop_output_types_func(env: _EvalEnv) -> list[TypeProto]:
    """Evaluate Loop output types from the body graph and carried inputs."""
    body_attr = env.attribute_protos.get("body")
    if body_attr is None:
        return []
    result: list[TypeProto] = []
    carried_input_types = env.node_input_types[2:]
    output_count = max(0, len(body_attr.g.output) - 1)
    for index in range(output_count):
        body_index = index + 1
        output_tp = copy.deepcopy(body_attr.g.output[body_index].type)
        existing_output_tp = (
            env.node_output_types[index]
            if index < len(env.node_output_types)
            else None
        )
        if index < len(carried_input_types) and carried_input_types[index] is not None:
            carried_input_tp = carried_input_types[index]
            if (
                carried_input_tp.HasField("optional_type")
                and existing_output_tp is not None
                and existing_output_tp.HasField("sequence_type")
                and existing_output_tp.sequence_type.elem_type.HasField("tensor_type")
                and not existing_output_tp.sequence_type.elem_type.tensor_type.HasField("shape")
            ):
                output_tp = copy.deepcopy(existing_output_tp)
            elif not (
                carried_input_tp.HasField("optional_type")
                and output_tp.HasField("sequence_type")
                and output_tp.sequence_type.elem_type.HasField("tensor_type")
                and not output_tp.sequence_type.elem_type.tensor_type.HasField("shape")
            ):
                carried_tp = _unwrap_optional_type_proto(carried_input_tp)
                output_tp = _merge_type_protos(output_tp, carried_tp)
        result.append(output_tp)
    return result


def _eval_attribute_func(call: FuncCall, env: _EvalEnv) -> Any:
    """Evaluate ``attribute(name, default)`` against runtime attributes."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else None
    return env.attributes.get(name, default)


def _eval_input_type_func(call: FuncCall, env: _EvalEnv) -> int:
    """Evaluate ``input_type(name, default)`` against runtime input types."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else TensorProto.UNDEFINED
    return env.elem_types.get(name, default)


def _eval_tensor_attribute_type_func(call: FuncCall, env: _EvalEnv) -> int:
    """Evaluate ``tensor_attribute_type(name, default)`` generically."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else TensorProto.UNDEFINED
    attr = env.attribute_protos.get(name)
    if attr is None or attr.type != onnx.AttributeProto.TENSOR:
        return default
    return attr.t.data_type


def _eval_tensor_attribute_shape_func(call: FuncCall, env: _EvalEnv) -> list[int | None]:
    """Evaluate ``tensor_attribute_shape(name, default)`` generically."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else []
    attr = env.attribute_protos.get(name)
    if attr is None or attr.type != onnx.AttributeProto.TENSOR:
        return _to_shape(default)
    return list(attr.t.dims)


def _eval_tensor_attribute_values_func(call: FuncCall, env: _EvalEnv) -> list[Any]:
    """Evaluate ``tensor_attribute_values(name, default)`` generically."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else []
    attr = env.attribute_protos.get(name)
    if attr is None or attr.type != onnx.AttributeProto.TENSOR:
        return list(default) if isinstance(default, (list, tuple)) else [default]
    values = _get_tensor_values(attr.t)
    if values is None:
        return list(default) if isinstance(default, (list, tuple)) else [default]
    return list(values)


def _eval_attribute_value_type_func(call: FuncCall, env: _EvalEnv) -> int:
    """Evaluate ``attribute_value_type(name, default)`` generically."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else TensorProto.UNDEFINED
    attr = env.attribute_protos.get(name)
    if attr is None:
        return default
    return _get_attribute_value_type(attr)


def _eval_attribute_value_shape_func(call: FuncCall, env: _EvalEnv) -> list[int | None]:
    """Evaluate ``attribute_value_shape(name, default)`` generically."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else []
    attr = env.attribute_protos.get(name)
    if attr is None:
        return _to_shape(default)
    shape = _get_attribute_value_shape(attr)
    if shape is None:
        return _to_shape(default)
    return shape


def _eval_attribute_values_func(call: FuncCall, env: _EvalEnv) -> list[Any]:
    """Evaluate ``attribute_values(name, default)`` generically."""
    name = _eval_expr(call.args[0], env)
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else []
    attr = env.attribute_protos.get(name)
    if attr is None:
        return list(default) if isinstance(default, (list, tuple)) else [default]
    values = _get_attribute_values(attr)
    if values is None:
        return list(default) if isinstance(default, (list, tuple)) else [default]
    return list(values)


def _eval_sequence_elem_shape_func(call: FuncCall, env: _EvalEnv) -> list[DimValue]:
    """Evaluate ``sequence_elem_shape(x)`` from a known sequence type."""
    arg = call.args[0]
    tp = _eval_full_type_func(arg, env)
    if tp is None or not tp.HasField("sequence_type"):
        return []
    elem_tp = tp.sequence_type.elem_type
    shape = _get_shape_from_type(elem_tp)
    return [] if shape is None else shape


def _eval_sequence_elem_type_func(call: FuncCall, env: _EvalEnv) -> int:
    """Evaluate ``sequence_elem_type(x, default)`` from a known sequence type."""
    arg = call.args[0]
    default = _eval_expr(call.args[1], env) if len(call.args) > 1 else TensorProto.UNDEFINED
    tp = _eval_full_type_func(arg, env)
    if tp is None or not tp.HasField("sequence_type"):
        return int(default)
    elem_tp = tp.sequence_type.elem_type
    elem_type = _get_elem_type(elem_tp)
    return elem_type if elem_type != TensorProto.UNDEFINED else int(default)


def _eval_func(call: FuncCall, env: _EvalEnv) -> Any:
    """Evaluate a function call."""
    name = call.name

    # Env-dependent special dispatch (these access env directly, not just args)
    special = _SPECIAL_FUNC_DISPATCH.get(name)
    if special is not None:
        return special(call, env)

    # shape_value needs special handling: resolve tensor values, not shape
    if name == "shape_value":
        arg_name = call.args[0]
        if isinstance(arg_name, Identifier) and arg_name.name in env.tensor_values:
            return list(env.tensor_values[arg_name.name])
        if isinstance(arg_name, Identifier) and arg_name.name in env.shapes:
            # ``shape_value`` resolves shape-tensor contents, not the tensor's
            # own shape. Without a known value we must degrade conservatively.
            return []
        if isinstance(arg_name, Identifier) and arg_name.name not in env.variables:
            return []
        # Fall back to evaluating normally
        args = [_eval_expr(a, env) for a in call.args]
        return _BUILTINS[name](args)

    # resolve_reshape: inject allowzero attribute from environment
    if name == "resolve_reshape":
        args = [_eval_expr(a, env) for a in call.args]
        if len(args) < 3:
            args.append(env.attributes.get("allowzero", 0))
        return _BUILTINS[name](args)

    # Regular built-in dispatch
    if name in _BUILTINS:
        args = [_eval_expr(a, env) for a in call.args]
        return _BUILTINS[name](args)

    raise NameError(f"Unknown function: {name!r}")


# Dispatch table for env-dependent special functions (avoids long if-chain).
_SPECIAL_FUNC_DISPATCH: dict[str, Any] = {
    "type": lambda call, env: _eval_type_func(call.args[0], env),
    "full_type": lambda call, env: _eval_full_type_func(call.args[0], env),
    "unwrap_optional_type": _eval_unwrap_optional_type_func,
    "if_output_types": lambda call, env: _eval_if_output_types_func(env),
    "loop_output_types": lambda call, env: _eval_loop_output_types_func(env),
    "output_count": lambda call, env: _eval_output_count_func(env),
    "attribute": _eval_attribute_func,
    "input_type": _eval_input_type_func,
    "tensor_attribute_type": _eval_tensor_attribute_type_func,
    "tensor_attribute_shape": _eval_tensor_attribute_shape_func,
    "tensor_attribute_values": _eval_tensor_attribute_values_func,
    "attribute_value_type": _eval_attribute_value_type_func,
    "attribute_value_shape": _eval_attribute_value_shape_func,
    "attribute_values": _eval_attribute_values_func,
    "sequence_elem_shape": _eval_sequence_elem_shape_func,
    "sequence_elem_type": _eval_sequence_elem_type_func,
}



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
    elem_types: dict[str, Any] | None = None,
    attribute_protos: dict[str, Any] | None = None,
    full_types: dict[str, Any] | None = None,
    node_input_types: list[Any] | None = None,
    node_output_types: list[Any] | None = None,
) -> tuple[dict[str, list[int | None]], dict[str, int], dict[str, list[Any]], dict[str, Any]]:
    """Execute an OTSL spec and return the output shapes and element types.

    Parameters
    ----------
    spec:
        Parsed OTSL shape specification.
    shapes:
        Mapping from input name (as declared in the spec) to its concrete shape.
    attributes:
        Mapping from attribute name to its value.
    tensor_values:
        Optional mapping from input name to its concrete integer values
        (needed for shape-tensor inputs like Reshape's ``shape`` input).
    elem_types:
        Optional mapping from input name to its ONNX element type
        (``TensorProto`` enum value).  For variadic inputs the value is a
        ``list[int]``.

    Returns
    -------
    tuple
        ``(shape_results, type_results, value_results, onnx_type_results)`` where *shape_results*
        maps output names to inferred shapes, *type_results* maps output names
        to inferred element types, *value_results* maps output names to
        flattened tensor values for shape-relevant outputs, and
        *onnx_type_results* maps output names to full ONNX type protos.
    """
    env = _EvalEnv(
        shapes,
        attributes,
        tensor_values,
        elem_types,
        attribute_protos,
        full_types,
        node_input_types,
        node_output_types,
    )
    shape_results: dict[str, list[int | None]] = {}
    type_results: dict[str, int] = {}
    value_results: dict[str, list[Any]] = {}
    onnx_type_results: dict[str, Any] = {}

    for stmt in spec.statements:
        if isinstance(stmt, LetStmt):
            env.variables[stmt.name] = _eval_expr(stmt.expr, env)

        elif isinstance(stmt, RequireStmt):
            val = _eval_expr(stmt.expr, env)
            if not val:
                raise ConstraintViolation(f"Constraint violated: {stmt.expr}")

        elif isinstance(stmt, ResultStmt):
            if stmt.field == "shape":
                val = _eval_expr(stmt.expr, env)
                shape_results[stmt.target] = _to_shape(val)
            elif stmt.field == "type":
                val = _eval_expr(stmt.expr, env)
                type_results[stmt.target] = int(val)
            elif stmt.field == "value":
                val = _eval_expr(stmt.expr, env)
                if isinstance(val, (list, tuple)):
                    value_results[stmt.target] = list(val)
                else:
                    value_results[stmt.target] = [val]
            elif stmt.field == "onnx_type":
                val = _eval_expr(stmt.expr, env)
                if isinstance(val, TypeProto):
                    onnx_type_results[stmt.target] = copy.deepcopy(val)
                elif isinstance(val, list) and all(isinstance(item, TypeProto) for item in val):
                    onnx_type_results[stmt.target] = list(val)  # type: ignore[assignment]

    return shape_results, type_results, value_results, onnx_type_results


# ---------------------------------------------------------------------------
# ONNX helper utilities
# ---------------------------------------------------------------------------

# Default attribute values for operators (used when the attribute is absent).
_DEFAULT_ATTRS: dict[str, dict[str, Any]] = {
    "Bernoulli": {"dtype": 0},
    "EyeLike": {"dtype": 0},
    "Flatten": {"axis": 1},
    "Gather": {"axis": 0},
    "GatherND": {"batch_dims": 0},
    "Gemm": {"transA": 0, "transB": 0},
    "QuantizeLinear": {"output_dtype": 0},
    "Softmax": {"axis": -1},
    "LogSoftmax": {"axis": -1},
    "Hardmax": {"axis": -1},
    "ArgMax": {"axis": 0, "keepdims": 1},
    "ArgMin": {"axis": 0, "keepdims": 1},
    "ReduceMax": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceMean": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceMin": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceProd": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceSum": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceLogSum": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceLogSumExp": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceL1": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceL2": {"keepdims": 1, "noop_with_empty_axes": 0},
    "ReduceSumSquare": {"keepdims": 1, "noop_with_empty_axes": 0},
    "AveragePool": {"strides": [], "pads": [], "dilations": [], "ceil_mode": 0, "auto_pad": "NOTSET"},
    "MaxPool": {"strides": [], "pads": [], "dilations": [], "ceil_mode": 0, "auto_pad": "NOTSET", "storage_order": 0},
    "LpPool": {"strides": [], "pads": [], "dilations": [], "ceil_mode": 0, "auto_pad": "NOTSET"},
    "Conv": {"strides": [], "pads": [], "dilations": [], "group": 1, "auto_pad": "NOTSET"},
    "ConvTranspose": {"strides": [], "pads": [], "dilations": [], "group": 1, "auto_pad": "NOTSET", "output_padding": [], "output_shape": []},
    "DepthToSpace": {"blocksize": 1},
    "SpaceToDepth": {"blocksize": 1},
    "Split": {"axis": 0, "num_outputs": 0},
    "OneHot": {"axis": -1},
    "TopK": {"axis": -1},
    "NegativeLogLikelihoodLoss": {"reduction": "mean"},
    "SoftmaxCrossEntropyLoss": {"reduction": "mean"},
    "RoiAlign": {"output_height": 1, "output_width": 1},
    "DFT": {"axis": 1, "onesided": 0},
    "STFT": {"onesided": 1},
}


def _get_shape_from_type(tp: TypeProto) -> list[DimValue] | None:
    """Extract a shape from an ONNX TypeProto."""
    if tp.HasField("optional_type"):
        return _get_shape_from_type(tp.optional_type.elem_type)
    if not tp.HasField("tensor_type"):
        return None
    tt = tp.tensor_type
    if not tt.HasField("shape"):
        return None
    dims: list[DimValue] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return dims


def _get_elem_type(tp: TypeProto) -> int:
    """Extract element type from ONNX TypeProto."""
    if tp.HasField("optional_type"):
        return _get_elem_type(tp.optional_type.elem_type)
    if tp.HasField("tensor_type"):
        return tp.tensor_type.elem_type
    return TensorProto.UNDEFINED


def _set_dim_proto(
    dim: onnx.TensorShapeProto.Dimension,
    value: DimValue,
    unknown_index: int,
) -> int:
    """Populate one ONNX dimension proto from a symbolic shape value."""
    if _is_known_int(value):
        dim.dim_value = value
        return unknown_index
    if isinstance(value, str):
        dim.dim_param = value
        return unknown_index
    return unknown_index


def _make_type_proto(
    shape: list[DimValue], elem_type: int = TensorProto.FLOAT
) -> TypeProto:
    """Create an ONNX TypeProto from a shape list."""
    tp = TypeProto()
    tensor_tp = tp.tensor_type
    tensor_tp.elem_type = elem_type
    shape_pb = tensor_tp.shape
    shape_pb.SetInParent()  # force creation even for 0-D scalars
    unknown_index = 0
    for d in shape:
        dim = shape_pb.dim.add()
        unknown_index = _set_dim_proto(dim, d, unknown_index)
    return tp


def _make_sequence_type_proto(
    shape: list[DimValue] | None,
    elem_type: int = TensorProto.FLOAT,
) -> TypeProto:
    """Create a sequence-of-tensor ONNX TypeProto."""
    tp = TypeProto()
    elem_tp = tp.sequence_type.elem_type.tensor_type
    elem_tp.elem_type = elem_type
    if shape is not None:
        shape_pb = elem_tp.shape
        shape_pb.SetInParent()
        unknown_index = 0
        for dim_value in shape:
            dim = shape_pb.dim.add()
            unknown_index = _set_dim_proto(dim, dim_value, unknown_index)
    return tp


def _merge_tensor_shapes(
    left: list[DimValue] | None,
    right: list[DimValue] | None,
) -> list[DimValue] | None:
    """Merge two tensor shapes conservatively."""
    if left is None:
        return right
    if right is None:
        return left
    if len(left) != len(right):
        return None
    merged: list[int | None] = []
    for l, r in zip(left, right):
        l_unknown = l is None or (isinstance(l, str) and l.startswith("unk__"))
        r_unknown = r is None or (isinstance(r, str) and r.startswith("unk__"))
        if l == r:
            merged.append(l)
        elif l_unknown:
            merged.append(r)
        elif r_unknown:
            merged.append(l)
        else:
            merged.append(None)
    return merged


def _merge_type_protos(left: TypeProto, right: TypeProto) -> TypeProto:
    """Merge two ONNX TypeProto instances conservatively."""
    if left.HasField("tensor_type") and right.HasField("tensor_type"):
        left_shape = _get_shape_from_type(left)
        right_shape = _get_shape_from_type(right)
        merged_shape = _merge_tensor_shapes(left_shape, right_shape)
        if merged_shape is None:
            return copy.deepcopy(left)
        elem_type = left.tensor_type.elem_type or right.tensor_type.elem_type
        return _make_type_proto(merged_shape, elem_type)

    if left.HasField("sequence_type") and right.HasField("sequence_type"):
        left_elem = left.sequence_type.elem_type
        right_elem = right.sequence_type.elem_type
        if left_elem.HasField("tensor_type") and right_elem.HasField("tensor_type"):
            merged_shape = _merge_tensor_shapes(
                _get_shape_from_type(left_elem),
                _get_shape_from_type(right_elem),
            )
            if merged_shape is None:
                return copy.deepcopy(left)
            elem_type = (
                left_elem.tensor_type.elem_type or right_elem.tensor_type.elem_type
            )
            return _make_sequence_type_proto(merged_shape, elem_type)

    return copy.deepcopy(left)


def _merge_or_set_shape(
    known_shapes: dict[str, list[int | None]],
    name: str,
    shape: list[int | None],
) -> None:
    """Preserve existing concrete shape information when new data is weaker."""
    if name in known_shapes:
        merged = _merge_tensor_shapes(known_shapes[name], shape)
        known_shapes[name] = shape if merged is None else merged
    else:
        known_shapes[name] = shape


def _merge_or_set_type(
    known_types: dict[str, TypeProto],
    name: str,
    tp: TypeProto,
) -> None:
    """Preserve existing concrete type information when new data is weaker."""
    if name in known_types:
        known_types[name] = _merge_type_protos(known_types[name], tp)
    else:
        known_types[name] = copy.deepcopy(tp)


def _get_initializer_values(initializer: TensorProto) -> list[int]:
    """Extract integer values from an ONNX initializer."""
    arr = numpy_helper.to_array(initializer)
    return arr.flatten().astype(int).tolist()


def _get_tensor_values(initializer: TensorProto) -> list[Any] | None:
    """Extract flattened tensor values when they are small and shape-relevant."""
    arr = numpy_helper.to_array(initializer)
    flat = arr.flatten()
    if flat.size > 64:
        return None
    values = flat.tolist()
    if isinstance(values, list):
        return values
    return [values]


def _get_constant_values(node_attr_map: dict[str, Any]) -> list[Any] | None:
    """Extract flattened Constant node payload values when available."""
    if "value" in node_attr_map:
        attr = node_attr_map["value"]
        if attr.type == onnx.AttributeProto.TENSOR:
            return _get_tensor_values(attr.t)
    if "value_int" in node_attr_map:
        return [node_attr_map["value_int"].i]
    if "value_float" in node_attr_map:
        return [node_attr_map["value_float"].f]
    if "value_ints" in node_attr_map:
        return list(node_attr_map["value_ints"].ints)
    if "value_floats" in node_attr_map:
        return list(node_attr_map["value_floats"].floats)
    return None


def _flatten_single_value(values: list[Any]) -> Any:
    """Return the only element from a scalar-like flattened tensor."""
    if len(values) != 1:
        raise ValueError("expected a single flattened tensor value")
    return values[0]


def _broadcast_values(left: list[Any], right: list[Any]) -> tuple[list[Any], list[Any]]:
    """Broadcast scalar flattened tensors against 1-D flattened tensors."""
    if len(left) == len(right):
        return left, right
    if len(left) == 1:
        return left * len(right), right
    if len(right) == 1:
        return left, right * len(left)
    raise ValueError("incompatible flattened tensor lengths")


_ELEMENTWISE_OPS: dict[str, Any] = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mul": operator.mul,
    "Div": lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) else a / b,
    "Max": max,
    "Min": min,
}


def _elementwise_values(
    left: list[Any],
    right: list[Any],
    op: str,
) -> list[Any]:
    """Apply a simple elementwise operation to flattened tensors."""
    fn = _ELEMENTWISE_OPS.get(op)
    if fn is None:
        raise ValueError(f"unsupported op {op!r}")
    left, right = _broadcast_values(left, right)
    return [fn(a, b) for a, b in zip(left, right)]


def _shape_tensor_values(shape: list[DimValue]) -> list[int] | None:
    """Convert a fully known shape into a concrete shape-tensor payload."""
    return None if any(not _is_known_int(d) for d in shape) else [int(d) for d in shape]


def _infer_tensor_value_output(
    node: Any,
    known_shapes: dict[str, list[int | None]],
    known_tensor_values: dict[str, list[Any]],
    attrs: dict[str, Any],
    node_attr_map: dict[str, Any],
) -> list[Any] | None:
    """Infer flattened tensor values for shape-relevant helper tensors."""
    op_type = node.op_type

    if op_type == "Identity" and len(node.input) >= 1 and node.input[0] in known_tensor_values:
        return list(known_tensor_values[node.input[0]])

    if op_type == "Cast" and len(node.input) >= 1 and node.input[0] in known_tensor_values:
        return list(known_tensor_values[node.input[0]])

    if op_type == "ConstantOfShape" and node.input and node.input[0] in known_tensor_values:
        dims = [int(v) for v in known_tensor_values[node.input[0]]]
        if any(dim < 0 for dim in dims):
            return None
        size = 1
        for dim in dims:
            size *= dim
        if size > 64:
            return None
        fill_value = 0
        values = _get_constant_values(node_attr_map)
        if values:
            fill_value = values[0]
        return [fill_value] * size

    if op_type == "Gather" and len(node.input) >= 2:
        data_name, indices_name = node.input[:2]
        if data_name in known_tensor_values and indices_name in known_tensor_values:
            axis = attrs.get("axis", 0)
            if axis != 0:
                return None
            data = known_tensor_values[data_name]
            result: list[Any] = []
            for raw_index in known_tensor_values[indices_name]:
                index = int(raw_index)
                if index < 0:
                    index += len(data)
                result.append(data[index])
            return result

    if op_type in _ELEMENTWISE_OPS and len(node.input) >= 2:
        left_name, right_name = node.input[:2]
        if left_name in known_tensor_values and right_name in known_tensor_values:
            return _elementwise_values(
                known_tensor_values[left_name],
                known_tensor_values[right_name],
                op_type,
            )

    if op_type == "Concat":
        axis = attrs.get("axis", 0)
        if axis != 0:
            return None
        result: list[Any] = []
        for name in node.input:
            if name not in known_tensor_values:
                return None
            result.extend(known_tensor_values[name])
        return result

    if op_type == "Slice" and node.input:
        data_name = node.input[0]
        if data_name not in known_tensor_values:
            return None
        data = known_tensor_values[data_name]
        starts_name = node.input[1] if len(node.input) >= 2 else ""
        ends_name = node.input[2] if len(node.input) >= 3 else ""
        axes_name = node.input[3] if len(node.input) >= 4 else ""
        steps_name = node.input[4] if len(node.input) >= 5 else ""
        if starts_name not in known_tensor_values or ends_name not in known_tensor_values:
            return None
        starts = known_tensor_values[starts_name]
        ends = known_tensor_values[ends_name]
        axes = known_tensor_values[axes_name] if axes_name in known_tensor_values else [0]
        steps = known_tensor_values[steps_name] if steps_name in known_tensor_values else [1]
        if len(axes) != 1 or int(axes[0]) != 0:
            return None
        start = int(_flatten_single_value(starts))
        end = int(_flatten_single_value(ends))
        step = int(_flatten_single_value(steps))
        return list(data[start:end:step])

    return None


def _get_attribute_value(attr: onnx.AttributeProto) -> Any:
    """Extract a Python value from an ONNX AttributeProto."""
    _atype = attr.type
    if _atype == onnx.AttributeProto.INT:
        return attr.i
    if _atype == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if _atype == onnx.AttributeProto.FLOAT:
        return attr.f
    if _atype == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if _atype == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if _atype == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    return None


def _get_attribute_value_shape(attr: onnx.AttributeProto) -> list[int | None] | None:
    """Return the tensor-like shape of an ONNX attribute value."""
    _atype = attr.type
    if _atype == onnx.AttributeProto.TENSOR:
        return list(attr.t.dims)
    if _atype in (onnx.AttributeProto.INT, onnx.AttributeProto.FLOAT, onnx.AttributeProto.STRING):
        return []
    if _atype == onnx.AttributeProto.INTS:
        return [len(attr.ints)]
    if _atype == onnx.AttributeProto.FLOATS:
        return [len(attr.floats)]
    if _atype == onnx.AttributeProto.STRINGS:
        return [len(attr.strings)]
    return None


def _get_attribute_value_type(attr: onnx.AttributeProto) -> int:
    """Return the tensor element type represented by an ONNX attribute value."""
    _atype = attr.type
    if _atype == onnx.AttributeProto.TENSOR:
        return attr.t.data_type
    if _atype in (onnx.AttributeProto.INT, onnx.AttributeProto.INTS):
        return TensorProto.INT64
    if _atype in (onnx.AttributeProto.FLOAT, onnx.AttributeProto.FLOATS):
        return TensorProto.FLOAT
    if _atype in (onnx.AttributeProto.STRING, onnx.AttributeProto.STRINGS):
        return TensorProto.STRING
    return TensorProto.UNDEFINED


def _get_attribute_values(attr: onnx.AttributeProto) -> list[Any] | None:
    """Return flattened tensor-like values from an ONNX attribute."""
    if attr.type == onnx.AttributeProto.TENSOR:
        return _get_tensor_values(attr.t)
    value = _get_attribute_value(attr)
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    return [value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OtslNumericalShapeInferenceEngine:
    """Numerical shape inference engine backed by OTSL operator specifications.

    Usage::

        engine = OtslNumericalShapeInferenceEngine()
        inferred_model = engine.infer_shapes(model)
    """

    def __init__(self) -> None:
        self._specs = load_all_specs()

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
        known_types: dict[str, TypeProto] = {}

        for inp in graph.input:
            s = _get_shape_from_type(inp.type)
            if s is not None:
                known_shapes[inp.name] = s
            known_elem_types[inp.name] = _get_elem_type(inp.type)
            known_types[inp.name] = copy.deepcopy(inp.type)

        # Values from existing value_info
        for vi in graph.value_info:
            s = _get_shape_from_type(vi.type)
            if s is not None:
                known_shapes[vi.name] = s
            known_elem_types[vi.name] = _get_elem_type(vi.type)
            known_types[vi.name] = copy.deepcopy(vi.type)

        for out in graph.output:
            known_types.setdefault(out.name, copy.deepcopy(out.type))

        # Initialiser shapes and values ----------------------------------------
        initializer_values: dict[str, list[int | float]] = {}
        known_tensor_values: dict[str, list[Any]] = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            known_shapes[init.name] = list(arr.shape)
            known_elem_types.setdefault(init.name, init.data_type)
            known_types.setdefault(init.name, _make_type_proto(list(arr.shape), init.data_type))
            # Keep integer values for shape-tensor resolution
            if init.data_type in (
                TensorProto.INT32,
                TensorProto.INT64,
                TensorProto.INT16,
                TensorProto.INT8,
            ):
                initializer_values[init.name] = arr.flatten().astype(int).tolist()
            # Also extract float values for operators that need them
            # (e.g. Resize/Upsample scales, Range start/limit/delta).
            elif init.data_type in (
                TensorProto.FLOAT,
                TensorProto.DOUBLE,
                TensorProto.FLOAT16,
            ):
                flat = arr.flatten().tolist()
                # Skip tensors containing NaN / Inf values.
                if any(math.isnan(v) or math.isinf(v) for v in flat):
                    continue
                # Store integer-valued floats as int for compatibility
                initializer_values[init.name] = [
                    int(v) if v == int(v) else v for v in flat
                ]
            tensor_values = _get_tensor_values(init)
            if tensor_values is not None:
                known_tensor_values[init.name] = tensor_values

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
            input_shapes: dict[str, list[int | None]] = {}
            input_elem_types: dict[str, Any] = {}
            input_full_types: dict[str, TypeProto] = {
                name: copy.deepcopy(tp) for name, tp in known_types.items()
            }
            tensor_vals: dict[str, list[int]] = {}

            if spec.inputs and spec.inputs[0].variadic:
                # Variadic: all node inputs map to the variadic name
                var_name = spec.inputs[0].name
                var_shapes = []
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
                                input_shapes[inp_decl.name] = shape
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
                            tensor_vals[inp_decl.name] = list(known_tensor_values[onnx_name])

            # Map node attributes -> OTSL attribute names
            attrs: dict[str, Any] = {}
            # Start with default attribute values
            if node.op_type in _DEFAULT_ATTRS:
                attrs.update(_DEFAULT_ATTRS[node.op_type])
            node_attr_map = {a.name: a for a in node.attribute}
            for attr_name, attr in node_attr_map.items():
                attrs[attr_name] = _get_attribute_value(attr)
            for attr_name in spec.attributes:
                if attr_name in node_attr_map:
                    attrs[attr_name] = _get_attribute_value(node_attr_map[attr_name])
            # Execute the spec --------------------------------------------------
            try:
                node_input_types = [
                    copy.deepcopy(known_types[name]) if name in known_types else None
                    for name in node.input
                ]
                node_output_types = [
                    copy.deepcopy(known_types[name]) if name in known_types else None
                    for name in node.output
                ]
                output_shapes, output_types, output_values, output_onnx_types = _execute_spec(
                    spec, input_shapes, attrs, tensor_vals,
                    elem_types=input_elem_types,
                    attribute_protos=node_attr_map,
                    full_types=input_full_types,
                    node_input_types=node_input_types,
                    node_output_types=node_output_types,
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
                continue  # graceful degradation for unsupported edge cases

            # Store results back ------------------------------------------------
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
                        shape = _to_shape(inferred)
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

                        # Determine element type from spec or inherit from
                        # the first input as a fallback.
                        et = output_types.get(out_name, TensorProto.UNDEFINED)
                        if et == TensorProto.UNDEFINED and node.input:
                            et = known_elem_types.get(
                                node.input[0], TensorProto.UNDEFINED
                            )
                        known_elem_types[onnx_out] = et

                        # Update value_info or output
                        tp = _make_type_proto(inferred, et)
                        _merge_or_set_type(known_types, onnx_out, tp)
                        if onnx_out not in value_info_names:
                            vi = graph.value_info.add()
                            vi.name = onnx_out
                            vi.type.CopyFrom(tp)
                            value_info_names.add(onnx_out)
                    if out_name in output_values:
                        known_tensor_values[onnx_out] = list(output_values[out_name])

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
                        known_tensor_values[onnx_out] = tensor_values

        # Patch graph outputs with inferred shapes ----------------------------
        for out in graph.output:
            if out.name in known_types:
                out.type.CopyFrom(_merge_type_protos(out.type, known_types[out.name]))
                continue
            if out.name in known_shapes:
                inferred = known_shapes[out.name]
                et = known_elem_types.get(out.name, _get_elem_type(out.type))
                out.type.CopyFrom(_merge_type_protos(out.type, _make_type_proto(inferred, et)))

        return model

    # -----------------------------------------------------------------------
    # Special-case helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _handle_constant(
        node: Any, node_attr_map: dict[str, Any],
    ) -> tuple[list[int | None] | None, int]:
        """Infer shape and element type of a Constant node from its attribute."""
        if "value" in node_attr_map:
            attr = node_attr_map["value"]
            if attr.type == onnx.AttributeProto.TENSOR:
                t = attr.t
                shape = list(t.dims) if t.dims else []
                return shape, t.data_type
        if "value_int" in node_attr_map:
            return [], TensorProto.INT64
        if "value_float" in node_attr_map:
            return [], TensorProto.FLOAT
        if "value_ints" in node_attr_map:
            attr = node_attr_map["value_ints"]
            return [len(attr.ints)], TensorProto.INT64
        if "value_floats" in node_attr_map:
            attr = node_attr_map["value_floats"]
            return [len(attr.floats)], TensorProto.FLOAT
        if "value_string" in node_attr_map:
            return [], TensorProto.STRING
        if "value_strings" in node_attr_map:
            attr = node_attr_map["value_strings"]
            return [len(attr.strings)], TensorProto.STRING
        return None, TensorProto.UNDEFINED

    @staticmethod
    def _handle_split(
        node: Any,
        spec: ShapeSpec,
        input_shapes: dict[str, list[int | None]],
        attrs: dict[str, Any],
        tensor_vals: dict[str, list[int]],
        node_attr_map: dict[str, Any],
        known_shapes: dict[str, list[int | None]],
    ) -> dict[str, list[int | None]] | None:
        """Compute shapes for all Split outputs."""
        first_in = node.input[0] if node.input else ""
        if first_in not in known_shapes:
            return None
        in_shape = known_shapes[first_in]
        axis = attrs.get("axis", 0)
        rank = len(in_shape)
        if axis < 0:
            axis += rank

        # Get split sizes
        split_sizes: list[int] | None = None
        if "split" in node_attr_map:
            split_sizes = list(_get_attribute_value(node_attr_map["split"]))
        elif "split_sizes" in tensor_vals:
            split_sizes = tensor_vals["split_sizes"]
        elif len(node.input) >= 2 and node.input[1]:
            split_name = node.input[1]
            if split_name in tensor_vals:
                split_sizes = tensor_vals.get(split_name)

        num_outputs = len(node.output)
        if split_sizes is None:
            # Equal split
            dim = in_shape[axis]
            if dim is not None and num_outputs > 0:
                base = dim // num_outputs
                remainder = dim % num_outputs
                split_sizes = [base + (1 if i < remainder else 0) for i in range(num_outputs)]
            else:
                return None

        result: dict[str, list[int | None]] = {}
        for i, onnx_out in enumerate(node.output):
            if onnx_out and i < len(split_sizes):
                out_shape = list(in_shape)
                out_shape[axis] = split_sizes[i]
                result[onnx_out] = out_shape
        return result

    @staticmethod
    def _handle_resize_inputs(
        node: Any,
        input_shapes: dict[str, list[int | None]],
        tensor_vals: dict[str, list[int]],
        known_shapes: dict[str, list[int | None]],
        known_tensor_values: dict[str, list[Any]],
    ) -> None:
        """Prepare Resize inputs (scales or sizes) for spec evaluation."""
        # Resize inputs: X, roi, scales, sizes
        input_name = node.input[0] if node.input else ""
        input_shape = known_shapes.get(input_name, [])
        axes_attr = next(
            (_get_attribute_value(attr) for attr in node.attribute if attr.name == "axes"),
            None,
        )
        if len(node.input) >= 3 and node.input[2]:
            scales_name = node.input[2]
            if scales_name in known_tensor_values:
                scales = list(known_tensor_values[scales_name])
                if axes_attr and input_shape and len(scales) != len(input_shape):
                    expanded = [1] * len(input_shape)
                    for axis, scale in zip(axes_attr, scales):
                        expanded[int(axis)] = scale
                    scales = expanded
                tensor_vals["scales"] = scales
        if len(node.input) >= 4 and node.input[3]:
            sizes_name = node.input[3]
            if sizes_name in known_tensor_values:
                sizes = [int(v) for v in known_tensor_values[sizes_name]]
                if axes_attr and input_shape and len(sizes) != len(input_shape):
                    expanded = list(input_shape)
                    for axis, size in zip(axes_attr, sizes):
                        expanded[int(axis)] = size
                    sizes = expanded
                tensor_vals["sizes"] = sizes
        # Ensure both keys exist
        tensor_vals.setdefault("scales", [])
        tensor_vals.setdefault("sizes", [])

    @staticmethod
    def _handle_upsample_inputs(
        node: Any,
        tensor_vals: dict[str, list[int]],
        known_shapes: dict[str, list[int | None]],
        known_tensor_values: dict[str, list[Any]],
    ) -> None:
        """Prepare Upsample inputs for spec evaluation."""
        if len(node.input) >= 2 and node.input[1]:
            scales_name = node.input[1]
            if scales_name in known_tensor_values:
                tensor_vals["scales"] = list(known_tensor_values[scales_name])
        tensor_vals.setdefault("scales", [])
        tensor_vals.setdefault("sizes", [])


# Module-level convenience function matching ``onnx.shape_inference.infer_shapes``.
_DEFAULT_ENGINE = None


def infer_shapes(model: ModelProto) -> ModelProto:
    """Infer shapes for *model* using OTSL specifications.

    Drop-in replacement for ``onnx.shape_inference.infer_shapes``.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = OtslNumericalShapeInferenceEngine()
    return _DEFAULT_ENGINE.infer_shapes(model)
