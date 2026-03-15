"""OSCL — ONNX Shape Constraint Language reference implementation."""

from .loader import load_all_specs, load_spec
from .parser import parse

__all__ = ["parse", "load_spec", "load_all_specs"]

# Engine is importable but not auto-loaded to avoid onnx/numpy dependency
# for pure parsing use cases.  Use: ``from oscl.engine import infer_shapes``
