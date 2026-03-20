"""OTSL - ONNX Type and Shape Language reference implementation."""

from .loader import load_all_specs, load_spec
from .parser import parse

__all__ = ["parse", "load_spec", "load_all_specs"]

# The numerical engine is importable but not auto-loaded to avoid onnx/numpy
# dependency for pure parsing use cases. Use:
# ``from otsl.numerical_engine import infer_shapes``
