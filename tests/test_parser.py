"""Tests for the OTSL parser."""

from pathlib import Path

import pytest

from oscl.ast import (
    BinOp,
    FuncCall,
    Identifier,
    IfExpr,
    IndexExpr,
    InputDecl,
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
from oscl.parser import ParseError, parse


# ---------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------

class TestBasicParsing:
    def test_empty_shape(self) -> None:
        spec = parse("shape { }")
        assert spec == ShapeSpec()

    def test_inputs_outputs(self) -> None:
        spec = parse("shape { inputs A, B; outputs Y; }")
        assert spec.inputs == [InputDecl("A"), InputDecl("B")]
        assert spec.outputs == ["Y"]

    def test_variadic_input(self) -> None:
        spec = parse("shape { inputs Xs[]; outputs Y; }")
        assert spec.inputs == [InputDecl("Xs", variadic=True)]

    def test_attributes(self) -> None:
        spec = parse("shape { attributes axis, perm; }")
        assert spec.attributes == ["axis", "perm"]


# ---------------------------------------------------------------
# Statements
# ---------------------------------------------------------------

class TestStatements:
    def test_require(self) -> None:
        spec = parse("shape { require dim(A,-1) == dim(B,-2); }")
        stmt = spec.statements[0]
        assert isinstance(stmt, RequireStmt)
        assert isinstance(stmt.expr, BinOp)
        assert stmt.expr.op == "=="

    def test_let(self) -> None:
        spec = parse("shape { let x = dim(A,0); }")
        stmt = spec.statements[0]
        assert isinstance(stmt, LetStmt)
        assert stmt.name == "x"
        assert isinstance(stmt.expr, FuncCall)

    def test_result(self) -> None:
        spec = parse("shape { outputs Y; result Y = shape(X); }")
        stmt = spec.statements[0]
        assert isinstance(stmt, ResultStmt)
        assert stmt.name == "Y"

    def test_when(self) -> None:
        src = """
        shape {
            when rank(A) == 1 {
                result Y = shape(A);
            }
        }
        """
        spec = parse(src)
        stmt = spec.statements[0]
        assert isinstance(stmt, WhenStmt)
        assert isinstance(stmt.condition, BinOp)
        assert len(stmt.body) == 1


# ---------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------

class TestExpressions:
    def test_number(self) -> None:
        spec = parse("shape { let x = 42; }")
        assert isinstance(spec.statements[0].expr, NumberLit)
        assert spec.statements[0].expr.value == 42

    def test_negative_number(self) -> None:
        spec = parse("shape { let x = dim(A,-1); }")
        call = spec.statements[0].expr
        assert isinstance(call, FuncCall)
        assert isinstance(call.args[1], NumberLit)
        assert call.args[1].value == -1

    def test_unknown_dim(self) -> None:
        spec = parse("shape { let x = ?; }")
        assert isinstance(spec.statements[0].expr, UnknownDim)

    def test_string_literal(self) -> None:
        spec = parse('shape { let x = "N"; }')
        assert isinstance(spec.statements[0].expr, StringLit)
        assert spec.statements[0].expr.value == "N"

    def test_func_call(self) -> None:
        spec = parse("shape { let x = broadcast(shape(A), shape(B)); }")
        call = spec.statements[0].expr
        assert isinstance(call, FuncCall)
        assert call.name == "broadcast"
        assert len(call.args) == 2

    def test_shape_literal(self) -> None:
        spec = parse("shape { let x = [dim(A,0), dim(B,1)]; }")
        lit = spec.statements[0].expr
        assert isinstance(lit, ShapeLiteral)
        assert len(lit.dims) == 2

    def test_index_expr(self) -> None:
        spec = parse("shape { let x = Xs[0]; }")
        idx = spec.statements[0].expr
        assert isinstance(idx, IndexExpr)
        assert isinstance(idx.obj, Identifier)
        assert idx.obj.name == "Xs"

    def test_if_expr(self) -> None:
        spec = parse("shape { let x = if transA then dim(A,-1) else dim(A,-2); }")
        expr = spec.statements[0].expr
        assert isinstance(expr, IfExpr)

    def test_map_expr(self) -> None:
        spec = parse("shape { let x = map i in Xs: dim(i,0); }")
        expr = spec.statements[0].expr
        assert isinstance(expr, MapExpr)
        assert expr.var == "i"

    def test_binary_ops(self) -> None:
        spec = parse("shape { require a + b == c * d; }")
        expr = spec.statements[0].expr
        assert isinstance(expr, BinOp)
        assert expr.op == "=="

    def test_logical_ops(self) -> None:
        spec = parse("shape { require a == 1 and b == 2 or c == 3; }")
        expr = spec.statements[0].expr
        # or has lowest precedence: (a==1 and b==2) or (c==3)
        assert isinstance(expr, BinOp)
        assert expr.op == "or"
        assert isinstance(expr.left, BinOp)
        assert expr.left.op == "and"

    def test_parenthesised_expr(self) -> None:
        spec = parse("shape { let x = (a + b) * c; }")
        expr = spec.statements[0].expr
        assert isinstance(expr, BinOp)
        assert expr.op == "*"
        assert isinstance(expr.left, BinOp)
        assert expr.left.op == "+"

    def test_arithmetic_addition(self) -> None:
        spec = parse("shape { let x = ax + 1; }")
        expr = spec.statements[0].expr
        assert isinstance(expr, BinOp)
        assert expr.op == "+"


# ---------------------------------------------------------------
# RFC example specs
# ---------------------------------------------------------------

class TestRFCExamples:
    def test_matmul(self) -> None:
        src = """
        shape {
          inputs A, B;
          outputs Y;
          require dim(A,-1) == dim(B,-2);
          let batch = broadcast(prefix(A,-2), prefix(B,-2));
          result Y = concat(batch, [dim(A,-2), dim(B,-1)]);
        }
        """
        spec = parse(src)
        assert len(spec.inputs) == 2
        assert spec.outputs == ["Y"]
        assert len(spec.statements) == 3

    def test_concat(self) -> None:
        src = """
        shape {
          inputs Xs[];
          outputs Y;
          attributes axis;
          let ax = normalize_axis(axis, rank(Xs[0]));
          result Y =
            map j in range(rank(Xs[0])):
              if j == ax
                then sum(map i in Xs: dim(i,j))
                else dim(Xs[0],j);
        }
        """
        spec = parse(src)
        assert spec.inputs[0].variadic is True
        result_stmt = spec.statements[1]
        assert isinstance(result_stmt, ResultStmt)
        assert isinstance(result_stmt.expr, MapExpr)

    def test_reshape(self) -> None:
        src = """
        shape {
          inputs data, shape_input;
          outputs reshaped;
          let target = shape_value(shape_input);
          result reshaped = resolve_reshape(shape(data), target);
        }
        """
        spec = parse(src)
        assert len(spec.inputs) == 2
        assert spec.outputs == ["reshaped"]

    def test_nonzero(self) -> None:
        src = """
        shape {
          inputs X;
          outputs Y;
          result Y = [rank(X), unknown_nonnegative()];
        }
        """
        spec = parse(src)
        result_stmt = spec.statements[0]
        assert isinstance(result_stmt, ResultStmt)
        assert isinstance(result_stmt.expr, ShapeLiteral)

    def test_gemm(self) -> None:
        src = """
        shape {
          inputs A, B, C;
          outputs Y;
          attributes transA, transB;
          let m = if transA then dim(A,-1) else dim(A,-2);
          let k1 = if transA then dim(A,-2) else dim(A,-1);
          let k2 = if transB then dim(B,-1) else dim(B,-2);
          let n = if transB then dim(B,-2) else dim(B,-1);
          require k1 == k2;
          result Y = [m, n];
        }
        """
        spec = parse(src)
        assert len(spec.statements) == 6


# ---------------------------------------------------------------
# All shipped spec files parse successfully
# ---------------------------------------------------------------

_SPECS_DIR = Path(__file__).resolve().parent.parent / "oscl" / "specs"


@pytest.mark.parametrize(
    "spec_file",
    sorted(_SPECS_DIR.glob("*.oscl")),
    ids=lambda p: p.stem,
)
def test_all_spec_files_parse(spec_file: Path) -> None:
    src = spec_file.read_text(encoding="utf-8")
    spec = parse(src)
    assert isinstance(spec, ShapeSpec)
    assert len(spec.outputs) >= 1


# ---------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------

class TestParseErrors:
    def test_missing_lbrace(self) -> None:
        with pytest.raises(ParseError):
            parse("shape }")

    def test_missing_semicolon(self) -> None:
        with pytest.raises(ParseError):
            parse("shape { let x = 1 }")

    def test_unexpected_token(self) -> None:
        with pytest.raises(ParseError):
            parse("shape { 42; }")

    def test_error_has_position(self) -> None:
        with pytest.raises(ParseError) as exc_info:
            parse("shape { let x = ; }")
        assert exc_info.value.line >= 1
        assert exc_info.value.col >= 1


# ---------------------------------------------------------------
# Comments
# ---------------------------------------------------------------

class TestComments:
    def test_hash_comment(self) -> None:
        src = """
        # This is a comment
        shape {
          inputs A; # inline comment
          outputs Y;
        }
        """
        spec = parse(src)
        assert spec.inputs == [InputDecl("A")]

    def test_slash_comment(self) -> None:
        src = """
        // This is a comment
        shape {
          inputs A; // inline comment
          outputs Y;
        }
        """
        spec = parse(src)
        assert spec.inputs == [InputDecl("A")]
