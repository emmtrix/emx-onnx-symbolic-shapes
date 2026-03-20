"""Recursive-descent parser for OTSL textual syntax."""

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
    NumberLit,
    RequireStmt,
    ResultStmt,
    ShapeLiteral,
    ShapeSpec,
    Statement,
    StringLit,
    UnknownDim,
)
from .lexer import Token, TokenType, tokenize

__all__ = ["ParseError", "parse"]

# Comparison token -> operator string (hoisted to module level).
_CMP_OPS: dict[TokenType, str] = {
    TokenType.EQ: "==", TokenType.NEQ: "!=",
    TokenType.LT: "<", TokenType.GT: ">",
    TokenType.LTE: "<=", TokenType.GTE: ">=",
}


class ParseError(Exception):
    """Raised when the parser encounters unexpected input."""

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Parse error at {line}:{col}: {message}")


class _Parser:
    """Internal recursive-descent parser state."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, tt: TokenType) -> bool:
        return self.tokens[self.pos].type == tt

    def _expect(self, tt: TokenType) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})",
                tok.line,
                tok.col,
            )
        self.pos += 1
        return tok

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    # -----------------------------------------------------------------
    # Top level
    # -----------------------------------------------------------------

    def parse_spec(self) -> ShapeSpec:
        self._expect(TokenType.RULES)
        self._expect(TokenType.LBRACE)

        inputs: list[InputDecl] = []
        outputs: list[str] = []
        attributes: list[str] = []
        statements: list[Statement] = []

        while not self._peek(TokenType.RBRACE) and not self._peek(TokenType.EOF):
            if self._peek(TokenType.INPUTS):
                inputs.extend(self._parse_input_decl())
            elif self._peek(TokenType.OUTPUTS):
                outputs.extend(self._parse_name_list(TokenType.OUTPUTS))
            elif self._peek(TokenType.ATTRIBUTES):
                attributes.extend(self._parse_name_list(TokenType.ATTRIBUTES))
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
        if not self._peek(TokenType.SEMICOLON):
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

    def _parse_name_list(self, keyword: TokenType) -> list[str]:
        """Parse ``keyword name, name, ...; `` (used for outputs and attributes)."""
        self._expect(keyword)
        names = [self._expect(TokenType.IDENT).value]
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
        target = self._expect(TokenType.IDENT).value
        self._expect(TokenType.DOT)
        field = self._expect(TokenType.IDENT).value
        self._expect(TokenType.ASSIGN)
        expr = self._parse_expr()
        self._expect(TokenType.SEMICOLON)
        return ResultStmt(target, field, expr)

    # -----------------------------------------------------------------
    # Expression grammar (precedence climbing)
    # -----------------------------------------------------------------

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._peek(TokenType.OR):
            self._advance()
            left = BinOp("or", left, self._parse_and())
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_comparison()
        while self._peek(TokenType.AND):
            self._advance()
            left = BinOp("and", left, self._parse_comparison())
        return left

    def _parse_comparison(self) -> Expr:
        left = self._parse_add()
        op = _CMP_OPS.get(self._cur().type)
        if op is not None:
            self._advance()
            left = BinOp(op, left, self._parse_add())
        return left

    def _parse_add(self) -> Expr:
        left = self._parse_mul()
        while self._cur().type in (TokenType.PLUS, TokenType.MINUS):
            op = "+" if self._cur().type == TokenType.PLUS else "-"
            self._advance()
            left = BinOp(op, left, self._parse_mul())
        return left

    def _parse_mul(self) -> Expr:
        left = self._parse_primary()
        while self._peek(TokenType.STAR):
            self._advance()
            left = BinOp("*", left, self._parse_primary())
        return left

    # -----------------------------------------------------------------
    # Primary expressions
    # -----------------------------------------------------------------

    def _parse_primary(self) -> Expr:
        tok = self._cur()

        if tok.type == TokenType.NUMBER:
            self._advance()
            return NumberLit(int(tok.value))

        if tok.type == TokenType.QUESTION:
            self._advance()
            return UnknownDim()

        if tok.type == TokenType.STRING:
            self._advance()
            return StringLit(tok.value)

        if tok.type == TokenType.LBRACKET:
            return self._parse_shape_literal()

        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        if tok.type == TokenType.IF:
            return self._parse_if_expr()

        if tok.type == TokenType.IDENT:
            return self._parse_ident_expr()

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
        return IfExpr(cond, then_expr, self._parse_expr())

    def _parse_ident_expr(self) -> Expr:
        name = self._expect(TokenType.IDENT).value

        # Function call: IDENT \'(\' ... \')\'
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

        # Index access: IDENT \'[\' expr \']\' 
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
    """Parse an OTSL source string and return a :class:`ShapeSpec` AST."""
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
