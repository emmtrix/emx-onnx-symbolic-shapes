"""AST node definitions for OSCL (ONNX Shape Constraint Language)."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Expr",
    "Statement",
    "NumberLit",
    "UnknownDim",
    "Identifier",
    "StringLit",
    "BinOp",
    "FuncCall",
    "ShapeLiteral",
    "IndexExpr",
    "IfExpr",
    "MapExpr",
    "RequireStmt",
    "LetStmt",
    "ResultStmt",
    "WhenStmt",
    "InputDecl",
    "ShapeSpec",
]


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    """Base class for all expression nodes."""


@dataclass
class NumberLit(Expr):
    """Integer literal (may be negative)."""
    value: int


@dataclass
class UnknownDim(Expr):
    """The ``?`` unknown‐dimension literal."""


@dataclass
class Identifier(Expr):
    """A bare name such as ``A``, ``axis``, or ``batch``."""
    name: str


@dataclass
class StringLit(Expr):
    """A quoted string literal, e.g. ``"N"``."""
    value: str


@dataclass
class BinOp(Expr):
    """Binary operation: arithmetic, comparison, or logical."""
    op: str
    left: Expr
    right: Expr


@dataclass
class FuncCall(Expr):
    """Function / operator call, e.g. ``dim(A, -1)``."""
    name: str
    args: list[Expr]


@dataclass
class ShapeLiteral(Expr):
    """Shape literal written as ``[expr, expr, ...]``."""
    dims: list[Expr]


@dataclass
class IndexExpr(Expr):
    """Index access, e.g. ``Xs[0]``."""
    obj: Expr
    index: Expr


@dataclass
class IfExpr(Expr):
    """Inline conditional: ``if cond then t else f``."""
    condition: Expr
    then_expr: Expr
    else_expr: Expr


@dataclass
class MapExpr(Expr):
    """Map comprehension: ``map var in iter: body``."""
    var: str
    iter_expr: Expr
    body: Expr


# ---------------------------------------------------------------------------
# Statement nodes
# ---------------------------------------------------------------------------

@dataclass
class Statement:
    """Base class for all statement nodes."""


@dataclass
class RequireStmt(Statement):
    """``require expr;``"""
    expr: Expr


@dataclass
class LetStmt(Statement):
    """``let name = expr;``"""
    name: str
    expr: Expr


@dataclass
class ResultStmt(Statement):
    """``result name = expr;``"""
    name: str
    expr: Expr


@dataclass
class WhenStmt(Statement):
    """``when condition { body }``"""
    condition: Expr
    body: list[Statement]


# ---------------------------------------------------------------------------
# Top‑level declarations and spec
# ---------------------------------------------------------------------------

@dataclass
class InputDecl:
    """An input declaration — ``name`` or ``name[]`` (variadic)."""
    name: str
    variadic: bool = False


@dataclass
class ShapeSpec:
    """Complete ``shape { … }`` specification."""
    inputs: list[InputDecl] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)
