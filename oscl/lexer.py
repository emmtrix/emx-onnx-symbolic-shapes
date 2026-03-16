"""Tokenizer for the OTSL textual syntax."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

__all__ = ["Token", "TokenType", "LexError", "tokenize"]


class TokenType(Enum):
    # Keywords
    SHAPE = auto()
    INPUTS = auto()
    OUTPUTS = auto()
    ATTRIBUTES = auto()
    REQUIRE = auto()
    LET = auto()
    RESULT = auto()
    WHEN = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    MAP = auto()
    IN = auto()
    AND = auto()
    OR = auto()

    # Delimiters / punctuation
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    COLON = auto()

    # Operators
    EQ = auto()       # ==
    NEQ = auto()      # !=
    LT = auto()       # <
    GT = auto()       # >
    LTE = auto()      # <=
    GTE = auto()      # >=
    ASSIGN = auto()   # =
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    QUESTION = auto()

    # Literals / identifiers
    NUMBER = auto()
    IDENT = auto()
    STRING = auto()

    # Sentinel
    EOF = auto()


_KEYWORDS: dict[str, TokenType] = {
    "shape": TokenType.SHAPE,
    "inputs": TokenType.INPUTS,
    "outputs": TokenType.OUTPUTS,
    "attributes": TokenType.ATTRIBUTES,
    "require": TokenType.REQUIRE,
    "let": TokenType.LET,
    "result": TokenType.RESULT,
    "when": TokenType.WHEN,
    "if": TokenType.IF,
    "then": TokenType.THEN,
    "else": TokenType.ELSE,
    "map": TokenType.MAP,
    "in": TokenType.IN,
    "and": TokenType.AND,
    "or": TokenType.OR,
}

# Token types after which a ``-`` followed by a digit is a negative number.
_NEG_NUM_PREDECESSORS = frozenset({
    TokenType.COMMA,
    TokenType.LPAREN,
    TokenType.LBRACKET,
    TokenType.ASSIGN,
    TokenType.PLUS,
    TokenType.MINUS,
    TokenType.STAR,
    TokenType.EQ,
    TokenType.NEQ,
    TokenType.LT,
    TokenType.GT,
    TokenType.LTE,
    TokenType.GTE,
    TokenType.COLON,
    # Keywords that precede an expression
    TokenType.REQUIRE,
    TokenType.WHEN,
    TokenType.IF,
    TokenType.THEN,
    TokenType.ELSE,
    TokenType.IN,
    TokenType.AND,
    TokenType.OR,
})


@dataclass(slots=True)
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:  # pragma: no cover
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


class LexError(Exception):
    """Raised on invalid input during tokenization."""

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Lex error at {line}:{col}: {message}")


def tokenize(source: str) -> list[Token]:
    """Tokenize an OTSL source string into a list of :class:`Token`."""
    tokens: list[Token] = []
    i = 0
    line = 1
    col = 1
    length = len(source)

    def _peek(offset: int = 0) -> str:
        pos = i + offset
        return source[pos] if pos < length else ""

    while i < length:
        ch = source[i]

        # --- whitespace ---
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue

        # --- comments (# or //) ---
        if ch == "#" or (ch == "/" and _peek(1) == "/"):
            while i < length and source[i] != "\n":
                i += 1
            continue

        start_col = col

        # --- string literal ---
        if ch == '"':
            j = i + 1
            while j < length and source[j] != '"':
                if source[j] == "\n":
                    raise LexError("Unterminated string literal", line, start_col)
                j += 1
            if j >= length:
                raise LexError("Unterminated string literal", line, start_col)
            value = source[i + 1 : j]
            tokens.append(Token(TokenType.STRING, value, line, start_col))
            col += j - i + 1
            i = j + 1
            continue

        # --- number literal ---
        if ch.isdigit():
            j = i
            while j < length and source[j].isdigit():
                j += 1
            value = source[i:j]
            tokens.append(Token(TokenType.NUMBER, value, line, start_col))
            col += j - i
            i = j
            continue

        # --- negative number (context-sensitive) ---
        if ch == "-" and _peek(1).isdigit():
            prev_type = tokens[-1].type if tokens else None
            if prev_type is None or prev_type in _NEG_NUM_PREDECESSORS:
                j = i + 1
                while j < length and source[j].isdigit():
                    j += 1
                value = source[i:j]
                tokens.append(Token(TokenType.NUMBER, value, line, start_col))
                col += j - i
                i = j
                continue

        # --- identifier / keyword ---
        if ch.isalpha() or ch == "_":
            j = i
            while j < length and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            tt = _KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line, start_col))
            col += j - i
            i = j
            continue

        # --- two-character operators ---
        two = source[i : i + 2] if i + 1 < length else ""
        if two == "==":
            tokens.append(Token(TokenType.EQ, "==", line, start_col))
            i += 2
            col += 2
            continue
        if two == "!=":
            tokens.append(Token(TokenType.NEQ, "!=", line, start_col))
            i += 2
            col += 2
            continue
        if two == "<=":
            tokens.append(Token(TokenType.LTE, "<=", line, start_col))
            i += 2
            col += 2
            continue
        if two == ">=":
            tokens.append(Token(TokenType.GTE, ">=", line, start_col))
            i += 2
            col += 2
            continue

        # --- single-character tokens ---
        singles: dict[str, TokenType] = {
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ";": TokenType.SEMICOLON,
            ",": TokenType.COMMA,
            ":": TokenType.COLON,
            "=": TokenType.ASSIGN,
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "?": TokenType.QUESTION,
            "<": TokenType.LT,
            ">": TokenType.GT,
        }
        if ch in singles:
            tokens.append(Token(singles[ch], ch, line, start_col))
            i += 1
            col += 1
            continue

        raise LexError(f"Unexpected character {ch!r}", line, start_col)

    tokens.append(Token(TokenType.EOF, "", line, col))
    return tokens
