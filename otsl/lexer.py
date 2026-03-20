"""Tokenizer for the OTSL textual syntax."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

__all__ = ["Token", "TokenType", "LexError", "tokenize"]


class TokenType(Enum):
    # Keywords
    RULES = auto()
    INPUTS = auto()
    OUTPUTS = auto()
    ATTRIBUTES = auto()
    REQUIRE = auto()
    LET = auto()
    RESULT = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
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
    DOT = auto()

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
    "rules": TokenType.RULES,
    "inputs": TokenType.INPUTS,
    "outputs": TokenType.OUTPUTS,
    "attributes": TokenType.ATTRIBUTES,
    "require": TokenType.REQUIRE,
    "let": TokenType.LET,
    "result": TokenType.RESULT,
    "if": TokenType.IF,
    "then": TokenType.THEN,
    "else": TokenType.ELSE,
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
    TokenType.IF,
    TokenType.THEN,
    TokenType.ELSE,
    TokenType.AND,
    TokenType.OR,
})

# Two-character operator lookup (hoisted to module level).
_TWO_CHAR_OPS: dict[str, TokenType] = {
    "==": TokenType.EQ,
    "!=": TokenType.NEQ,
    "<=": TokenType.LTE,
    ">=": TokenType.GTE,
}

# Single-character token lookup (hoisted to module level).
_SINGLE_CHAR: dict[str, TokenType] = {
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ";": TokenType.SEMICOLON,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
    ".": TokenType.DOT,
    "=": TokenType.ASSIGN,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "?": TokenType.QUESTION,
    "<": TokenType.LT,
    ">": TokenType.GT,
}


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
        if ch == "#" or (ch == "/" and i + 1 < length and source[i + 1] == "/"):
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
            tokens.append(Token(TokenType.STRING, source[i + 1 : j], line, start_col))
            col += j - i + 1
            i = j + 1
            continue

        # --- number literal ---
        if ch.isdigit():
            j = i
            while j < length and source[j].isdigit():
                j += 1
            tokens.append(Token(TokenType.NUMBER, source[i:j], line, start_col))
            col += j - i
            i = j
            continue

        # --- negative number (context-sensitive) ---
        if ch == "-" and i + 1 < length and source[i + 1].isdigit():
            prev_type = tokens[-1].type if tokens else None
            if prev_type is None or prev_type in _NEG_NUM_PREDECESSORS:
                j = i + 1
                while j < length and source[j].isdigit():
                    j += 1
                tokens.append(Token(TokenType.NUMBER, source[i:j], line, start_col))
                col += j - i
                i = j
                continue

        # --- identifier / keyword ---
        if ch.isalpha() or ch == "_":
            j = i
            while j < length and (source[j].isalnum() or source[j] == "_"):
                j += 1
            word = source[i:j]
            tokens.append(Token(_KEYWORDS.get(word, TokenType.IDENT), word, line, start_col))
            col += j - i
            i = j
            continue

        # --- two-character operators ---
        if i + 1 < length:
            two = source[i : i + 2]
            tt = _TWO_CHAR_OPS.get(two)
            if tt is not None:
                tokens.append(Token(tt, two, line, start_col))
                i += 2
                col += 2
                continue

        # --- single-character tokens ---
        tt = _SINGLE_CHAR.get(ch)
        if tt is not None:
            tokens.append(Token(tt, ch, line, start_col))
            i += 1
            col += 1
            continue

        raise LexError(f"Unexpected character {ch!r}", line, start_col)

    tokens.append(Token(TokenType.EOF, "", line, col))
    return tokens
