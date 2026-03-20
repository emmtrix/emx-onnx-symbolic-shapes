"""Utilities for loading ``.otsl`` spec files shipped with the package."""

from __future__ import annotations

from pathlib import Path

from .ast import ShapeSpec
from .parser import parse

__all__ = ["load_spec", "load_all_specs"]

_SPECS_DIR = Path(__file__).resolve().parent / "specs"


def load_spec(name: str) -> ShapeSpec:
    """Load a single operator spec by name (without extension).

    Example::

        spec = load_spec("matmul")
    """
    path = _SPECS_DIR / f"{name}.otsl"
    if not path.exists():
        raise FileNotFoundError(f"No spec file found for {name!r} at {path}")
    return parse(path.read_text(encoding="utf-8"))


def load_all_specs() -> dict[str, ShapeSpec]:
    """Load every ``.otsl`` file in the specs directory.

    Returns a mapping from operator name (stem of the filename) to the
    parsed :class:`ShapeSpec`.
    """
    return {
        p.stem: parse(p.read_text(encoding="utf-8"))
        for p in sorted(_SPECS_DIR.glob("*.otsl"))
    }
