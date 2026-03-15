"""Recursive‑descent parser for OSCL textual syntax."""

from __future__ import annotations

from .ast import (
    BinOp,
    Expr,
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
    Statement,
    StringLit,
    UnknownDim,
    WhenStmt,
)
from .lexer import Token, TokenType, tokenize

__all__ = ["ParseError", "parse"]


class ParseError(Exception):
    """Raised when the parser encounters unexpected input."""

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Parse error at {line}:{col}: {message}")


class _Parser:
    """Internal recursive‑descent parser state."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, tt: TokenType) -> bool:
        return self._cur().type == tt

    def _at_keyword(self, kw: TokenType) -> bool:
        return self._cur().type == kw

    def _expect(self, tt: TokenType) -> Token:
        tok = self._cur()
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})",
                tok.line,
                tok.col,
            )
        self.pos += 1
        return tok

    def _advance(self) -> Token:
        tok = self._cur()
        self.pos += 1
        return tok

    # -----------------------------------------------------------------
    # Top level
    # -----------------------------------------------------------------

    def parse_spec(self) -> ShapeSpec:
        self._expect(TokenType.SHAPE)
        self._expect(TokenType.LBRACE)

        inputs: list[InputDecl] = []
        outputs: list[str] = []
        attributes: list[str] = []
        statements: list[Statement] = []

        # Declarations first, then statements.  We allow them in any order,
        # but declarations are separated from general statements by type.
        while not self._peek(TokenType.RBRACE) and not self._peek(TokenType.EOF):
            if self._at_keyword(TokenType.INPUTS):
                inputs.extend(self._parse_input_decl())
            elif self._at_keyword(TokenType.OUTPUTS):
                outputs.extend(self._parse_output_decl())
            elif self._at_keyword(TokenType.ATTRIBUTES):
                attributes.extend(self._parse_attr_decl())
            else:
                statements.append(self._parse_statement())

        self._expect(TokenType.RBRACE)
        return ShapeSpec(inputs, outputs, attributes, statements)

    # -----------------------------------------------------------------
    # Declarations
    # -----------------------------------------------------------------

    def _parse_input_decl(self) -> list[InputDecl]:
        self._expect(TokenType.INPUTS)
        decls: list[InputDecl] = []
        decls.append(self._parse_one_input())
        while self._peek(TokenType.COMMA):
            self._advance()
            decls.append(self._parse_one_input())
        self._expect(TokenType.SEMICOLON)
        return decls

    def _parse_one_input(self) -> InputDecl:
        name_tok = self._expect(TokenType.IDENT)
        variadic = False
        if self._peek(TokenType.LBRACKET):
            self._advance()
            self._expect(TokenType.RBRACKET)
            variadic = True
        return InputDecl(name_tok.value, variadic)

    def _parse_output_decl(self) -> list[str]:
        self._expect(TokenType.OUTPUTS)
        names: list[str] = []
        names.append(self._expect(TokenType.IDENT).value)
        while self._peek(TokenType.COMMA):
            self._advance()
            names.append(self._expect(TokenType.IDENT).value)
        self._expect(TokenType.SEMICOLON)
        return names

    def _parse_attr_decl(self) -> list[str]:
        self._expect(TokenType.ATTRIBUTES)
        names: list[str] = []
        names.append(self._expect(TokenType.IDENT).value)
        while self._peek(TokenType.COMMA):
            self._advance()
            names.append(self._expect(TokenType.IDENT).value)
        self._expect(TokenType.SEMICOLON)
        return names

    # -----------------------------------------------------------------
    # Statements
    # -----------------------------------------------------------------

    def _parse_statement(self) -> Statement:
        tok = self._cur()
        if tok.type == TokenType.REQUIRE:
            return self._parse_require()
        if tok.type == TokenType.LET:
            return self._parse_let()
        if tok.type == TokenType.RESULT:
            return self._parse_result()
        if tok.type == TokenType.WHEN:
            return self._parse_when()
        raise ParseError(
            f"Expected statement, got {tok.type.name} ({tok.value!r})",
            tok.line,
            tok.col,
        )

    def _parse_require(self) -> RequireStmt:
        self._expect(TokenType.REQUIRE)
        expr = self._parse_expr()
        self._expect(TokenType.SEMICOLON)
        return RequireStmt(expr)

    def _parse_let(self) -> LetStmt:
        self._expect(TokenType.LET)
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.ASSIGN)
        expr = self._parse_expr()
        self._expect(TokenType.SEMICOLON)
        return LetStmt(name, expr)

    def _parse_result(self) -> ResultStmt:
        self._expect(TokenType.RESULT)
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.ASSIGN)
        expr = self._parse_expr()
        self._expect(TokenType.SEMICOLON)
        return ResultStmt(name, expr)

    def _parse_when(self) -> WhenStmt:
        self._expect(TokenType.WHEN)
        cond = self._parse_expr()
        self._expect(TokenType.LBRACE)
        body: list[Statement] = []
        while not self._peek(TokenType.RBRACE) and not self._peek(TokenType.EOF):
            body.append(self._parse_statement())
        self._expect(TokenType.RBRACE)
        return WhenStmt(cond, body)

    # -----------------------------------------------------------------
    # Expression grammar (precedence climbing)
    # -----------------------------------------------------------------

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._at_keyword(TokenType.OR):
            self._advance()
            right = self._parse_and()
            left = BinOp("or", left, right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_comparison()
        while self._at_keyword(TokenType.AND):
            self._advance()
            right = self._parse_comparison()
            left = BinOp("and", left, right)
        return left

    def _parse_comparison(self) -> Expr:
        left = self._parse_add()
        op_map = {
            TokenType.EQ: "==",
            TokenType.NEQ: "!=",
            TokenType.LT: "<",
            TokenType.GT: ">",
            TokenType.LTE: "<=",
            TokenType.GTE: ">=",
        }
        if self._cur().type in op_map:
            op = op_map[self._cur().type]
            self._advance()
            right = self._parse_add()
            left = BinOp(op, left, right)
        return left

    def _parse_add(self) -> Expr:
        left = self._parse_mul()
        while self._cur().type in (TokenType.PLUS, TokenType.MINUS):
            op = "+" if self._cur().type == TokenType.PLUS else "-"
            self._advance()
            right = self._parse_mul()
            left = BinOp(op, left, right)
        return left

    def _parse_mul(self) -> Expr:
        left = self._parse_primary()
        while self._cur().type == TokenType.STAR:
            self._advance()
            right = self._parse_primary()
            left = BinOp("*", left, right)
        return left

    # -----------------------------------------------------------------
    # Primary expressions
    # -----------------------------------------------------------------

    def _parse_primary(self) -> Expr:
        tok = self._cur()

        # Number
        if tok.type == TokenType.NUMBER:
            self._advance()
            return NumberLit(int(tok.value))

        # Unknown dimension
        if tok.type == TokenType.QUESTION:
            self._advance()
            return UnknownDim()

        # String literal
        if tok.type == TokenType.STRING:
            self._advance()
            return StringLit(tok.value)

        # Shape literal  [expr, ...]
        if tok.type == TokenType.LBRACKET:
            return self._parse_shape_literal()

        # Parenthesised expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        # if / then / else
        if tok.type == TokenType.IF:
            return self._parse_if_expr()

        # map var in iter: body
        if tok.type == TokenType.MAP:
            return self._parse_map_expr()

        # Identifier-led: plain ident, function call, or index access
        if tok.type == TokenType.IDENT:
            return self._parse_ident_expr()

        # 'shape' used as a function call (e.g., shape(X))
        if tok.type == TokenType.SHAPE and self.tokens[self.pos + 1].type == TokenType.LPAREN:
            self._advance()  # consume the SHAPE keyword
            self._advance()  # consume LPAREN
            args: list[Expr] = []
            if not self._peek(TokenType.RPAREN):
                args.append(self._parse_expr())
                while self._peek(TokenType.COMMA):
                    self._advance()
                    args.append(self._parse_expr())
            self._expect(TokenType.RPAREN)
            return FuncCall("shape", args)

        raise ParseError(
            f"Expected expression, got {tok.type.name} ({tok.value!r})",
            tok.line,
            tok.col,
        )

    def _parse_shape_literal(self) -> ShapeLiteral:
        self._expect(TokenType.LBRACKET)
        dims: list[Expr] = []
        if not self._peek(TokenType.RBRACKET):
            dims.append(self._parse_expr())
            while self._peek(TokenType.COMMA):
                self._advance()
                dims.append(self._parse_expr())
        self._expect(TokenType.RBRACKET)
        return ShapeLiteral(dims)

    def _parse_if_expr(self) -> IfExpr:
        self._expect(TokenType.IF)
        cond = self._parse_expr()
        self._expect(TokenType.THEN)
        then_expr = self._parse_expr()
        self._expect(TokenType.ELSE)
        else_expr = self._parse_expr()
        return IfExpr(cond, then_expr, else_expr)

    def _parse_map_expr(self) -> MapExpr:
        self._expect(TokenType.MAP)
        var = self._expect(TokenType.IDENT).value
        self._expect(TokenType.IN)
        iter_expr = self._parse_primary()
        self._expect(TokenType.COLON)
        body = self._parse_expr()
        return MapExpr(var, iter_expr, body)

    def _parse_ident_expr(self) -> Expr:
        name_tok = self._expect(TokenType.IDENT)
        name = name_tok.value

        # Function call: IDENT '(' ... ')'
        if self._peek(TokenType.LPAREN):
            self._advance()
            args: list[Expr] = []
            if not self._peek(TokenType.RPAREN):
                args.append(self._parse_expr())
                while self._peek(TokenType.COMMA):
                    self._advance()
                    args.append(self._parse_expr())
            self._expect(TokenType.RPAREN)
            return FuncCall(name, args)

        # Index access: IDENT '[' expr ']'
        if self._peek(TokenType.LBRACKET):
            self._advance()
            index = self._parse_expr()
            self._expect(TokenType.RBRACKET)
            return IndexExpr(Identifier(name), index)

        return Identifier(name)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def parse(source: str) -> ShapeSpec:
    """Parse an OSCL source string and return a :class:`ShapeSpec` AST."""
    tokens = tokenize(source)
    parser = _Parser(tokens)
    spec = parser.parse_spec()
    # Ensure we consumed everything
    if not parser._peek(TokenType.EOF):
        tok = parser._cur()
        raise ParseError(
            f"Unexpected trailing token {tok.type.name} ({tok.value!r})",
            tok.line,
            tok.col,
        )
    return spec
