"""Tests for the OSCL lexer."""

from oscl.lexer import LexError, Token, TokenType, tokenize
import pytest


class TestBasicTokenization:
    def test_empty_shape(self) -> None:
        tokens = tokenize("shape { }")
        types = [t.type for t in tokens]
        assert types == [TokenType.SHAPE, TokenType.LBRACE, TokenType.RBRACE, TokenType.EOF]

    def test_keywords(self) -> None:
        src = "shape inputs outputs attributes require let result when if then else map in and or"
        tokens = tokenize(src)
        expected = [
            TokenType.SHAPE, TokenType.INPUTS, TokenType.OUTPUTS,
            TokenType.ATTRIBUTES, TokenType.REQUIRE, TokenType.LET,
            TokenType.RESULT, TokenType.WHEN, TokenType.IF, TokenType.THEN,
            TokenType.ELSE, TokenType.MAP, TokenType.IN, TokenType.AND,
            TokenType.OR, TokenType.EOF,
        ]
        assert [t.type for t in tokens] == expected

    def test_identifiers_and_numbers(self) -> None:
        tokens = tokenize("A 42 hello_world")
        assert tokens[0] == Token(TokenType.IDENT, "A", 1, 1)
        assert tokens[1] == Token(TokenType.NUMBER, "42", 1, 3)
        assert tokens[2] == Token(TokenType.IDENT, "hello_world", 1, 6)

    def test_operators(self) -> None:
        tokens = tokenize("== != < > <= >= = + - * ?")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert types == [
            TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT,
            TokenType.LTE, TokenType.GTE, TokenType.ASSIGN, TokenType.PLUS,
            TokenType.MINUS, TokenType.STAR, TokenType.QUESTION,
        ]

    def test_delimiters(self) -> None:
        tokens = tokenize("{ } ( ) [ ] ; , :")
        types = [t.type for t in tokens[:-1]]
        assert types == [
            TokenType.LBRACE, TokenType.RBRACE, TokenType.LPAREN,
            TokenType.RPAREN, TokenType.LBRACKET, TokenType.RBRACKET,
            TokenType.SEMICOLON, TokenType.COMMA, TokenType.COLON,
        ]

    def test_string_literal(self) -> None:
        tokens = tokenize('"hello"')
        assert tokens[0] == Token(TokenType.STRING, "hello", 1, 1)


class TestNegativeNumbers:
    def test_negative_in_func_call(self) -> None:
        tokens = tokenize("dim(A,-1)")
        # Should be: IDENT LPAREN IDENT COMMA NUMBER RPAREN
        assert tokens[4].type == TokenType.NUMBER
        assert tokens[4].value == "-1"

    def test_negative_in_list(self) -> None:
        tokens = tokenize("[-2, 3]")
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "-2"

    def test_minus_in_expression(self) -> None:
        tokens = tokenize("a - 1")
        assert tokens[1].type == TokenType.MINUS
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "1"

    def test_negative_after_assign(self) -> None:
        tokens = tokenize("= -5")
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "-5"

    def test_negative_after_comparison(self) -> None:
        tokens = tokenize("== -3")
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "-3"


class TestComments:
    def test_hash_comment(self) -> None:
        tokens = tokenize("shape # this is a comment\n{ }")
        types = [t.type for t in tokens]
        assert types == [TokenType.SHAPE, TokenType.LBRACE, TokenType.RBRACE, TokenType.EOF]

    def test_slash_comment(self) -> None:
        tokens = tokenize("shape // comment\n{ }")
        types = [t.type for t in tokens]
        assert types == [TokenType.SHAPE, TokenType.LBRACE, TokenType.RBRACE, TokenType.EOF]


class TestLineColumn:
    def test_multiline(self) -> None:
        tokens = tokenize("shape\n{\n}")
        assert tokens[0].line == 1
        assert tokens[1].line == 2
        assert tokens[2].line == 3


class TestLexErrors:
    def test_unexpected_char(self) -> None:
        with pytest.raises(LexError, match="Unexpected character"):
            tokenize("shape @")

    def test_unterminated_string(self) -> None:
        with pytest.raises(LexError, match="Unterminated string"):
            tokenize('"hello')
