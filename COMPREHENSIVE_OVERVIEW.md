# Comprehensive Repository Overview: emx-onnx-symbolic-shapes

## 1. DIRECTORY TREE (2 levels deep)

```
emx-onnx-symbolic-shapes/
├── .git/                          # Git repository
├── .github/
│   └── copilot-instructions.md    # AI agent guidance
├── .gitignore                     # Standard Python .gitignore
├── LICENSE                        # MIT License
├── README.md                      # Project overview
├── agents.md                      # Detailed implementation guidance for AI agents
├── docs/
│   └── rfc-oscl.md               # Full RFC specification (540 lines)
├── oscl/                          # Main package
│   ├── __init__.py               # Public API exports
│   ├── ast.py                    # AST node definitions (158 lines)
│   ├── lexer.py                  # Tokenizer (264 lines)
│   ├── loader.py                 # Spec file loader (37 lines)
│   ├── parser.py                 # Parser (391 lines)
│   └── specs/                    # Operator specification files (13 .oscl files)
│       ├── add.oscl
│       ├── concat.oscl
│       ├── flatten.oscl
│       ├── gather.oscl
│       ├── gemm.oscl
│       ├── matmul.oscl
│       ├── nonzero.oscl
│       ├── relu.oscl
│       ├── reshape.oscl
│       ├── softmax.oscl
│       ├── squeeze.oscl
│       ├── transpose.oscl
│       └── unsqueeze.oscl
├── pyproject.toml                # Project configuration
└── tests/                        # Test suite
    ├── __init__.py
    ├── test_lexer.py            # Lexer tests
    ├── test_parser.py           # Parser tests
    └── test_loader.py           # Spec loader tests
```

---

## 2. PROJECT CONFIGURATION (pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "oscl"
version = "0.1.0"
description = "Reference implementation of OSCL (ONNX Shape Constraint Language)"
requires-python = ">=3.10"
license = {text = "MIT"}

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.setuptools.packages.find]
include = ["oscl*"]

[tool.setuptools.package-data]
oscl = ["specs/*.oscl"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Key Points:**
- **Python 3.10+** required
- **No external dependencies** (zero runtime dependencies!)
- Dev dependency: pytest for testing
- Installs specs as package data
- Tests configured to run from `tests/` directory

---

## 3. REQUIREMENTS FILES

**Result:** None exist. This project has ZERO external runtime dependencies. All functionality is implemented purely in the Python standard library.

---

## 4. .GITHUB DIRECTORY

Contains only: `copilot-instructions.md` (references the main `agents.md` file)

---

## 5. PACKAGE INITIALIZATION (oscl/__init__.py)

```python
"""OSCL — ONNX Shape Constraint Language reference implementation."""

from .loader import load_all_specs, load_spec
from .parser import parse

__all__ = ["parse", "load_spec", "load_all_specs"]
```

**Public API:**
- `parse(source: str) -> ShapeSpec` — Parse OSCL text syntax into AST
- `load_spec(name: str) -> ShapeSpec` — Load a single bundled spec file
- `load_all_specs() -> dict[str, ShapeSpec]` — Load all bundled specs

---

## 6. OSCL SPEC FILES (13 total in oscl/specs/)

All operator shape constraint specifications:

1. **add.oscl** (83 bytes)
2. **concat.oscl** (235 bytes)
3. **flatten.oscl** (181 bytes)
4. **gather.oscl** (208 bytes)
5. **gemm.oscl** (320 bytes) — General matrix multiply
6. **matmul.oscl** (180 bytes)
7. **nonzero.oscl** (83 bytes)
8. **relu.oscl** (59 bytes)
9. **reshape.oscl** (159 bytes)
10. **softmax.oscl** (77 bytes)
11. **squeeze.oscl** (153 bytes)
12. **transpose.oscl** (117 bytes)
13. **unsqueeze.oscl** (155 bytes)

---

## 7. SAMPLE SPEC FILES

### MatMul.oscl
```oscl
shape {
  inputs A, B;
  outputs Y;

  require dim(A,-1) == dim(B,-2);

  let batch = broadcast(prefix(A,-2), prefix(B,-2));

  result Y = concat(batch, [dim(A,-2), dim(B,-1)]);
}
```

**Key features:**
- Input/output declarations
- Type constraints (require statements)
- Intermediate computations (let bindings)
- Output shape computation (result statements)

### Concat.oscl
```oscl
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
```

**Key features:**
- **Variadic inputs** (`Xs[]`)
- Attribute handling
- **Map expressions** for shape comprehension
- **Conditional expressions** (if/then/else)

### Reshape.oscl
```oscl
shape {
  inputs data, shape_input;
  outputs reshaped;

  let target = shape_value(shape_input);

  result reshaped = resolve_reshape(shape(data), target);
}
```

### Gemm.oscl (General Matrix Multiply)
```oscl
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
```

### Add.oscl
```oscl
shape {
  inputs A, B;
  outputs C;

  result C = broadcast(shape(A), shape(B));
}
```

### ReLU.oscl
```oscl
shape {
  inputs X;
  outputs Y;

  result Y = shape(X);
}
```

---

## 8. TEST FILES

### tests/test_parser.py (331 lines)
- **Test basic structure:** empty specs, inputs/outputs, variadic inputs, attributes
- **Test statements:** require, let, result, when
- **Test expressions:** numbers, identifiers, strings, function calls, shape literals, index expressions, if expressions, map expressions, binary ops, logical ops
- **RFC examples:** Tests for matmul, concat, reshape, nonzero, gemm
- **Parametrized tests:** All 13 spec files are tested to ensure they parse successfully
- **Error handling:** Tests for missing braces, semicolons, unexpected tokens, error position tracking
- **Comments:** Tests for both `#` and `//` comment syntax

### tests/test_lexer.py (111 lines)
- **Basic tokenization:** Tests empty specs, keywords, identifiers, numbers, operators, delimiters, string literals
- **Negative numbers:** Tests context-aware negative number handling (after comma, equals, comparison operators, etc.)
- **Comments:** Tests `#` and `//` syntax
- **Line/column tracking:** Multiline position tracking
- **Error handling:** Tests for unexpected characters, unterminated strings

### tests/test_loader.py (46 lines)
- Tests loading individual specs (matmul, concat, relu)
- Tests error handling for nonexistent specs
- Tests loading all specs at once
- Verifies all specs have at least one output

---

## 9. .GITIGNORE

Standard comprehensive Python .gitignore covering:
- Python bytecode, eggs, distributions
- Virtual environments (.venv, venv, env)
- IDEs (PyCharm, VS Code)
- Testing artifacts (pytest, coverage, tox, nox)
- Build artifacts
- Documentation builds
- mypy, pytype, Pyre caches
- Project-specific tools (Abstra, Ruff, Cursor)

---

## 10. AST NODE DEFINITIONS (oscl/ast.py - First 50 lines)

```python
"""AST node definitions for OSCL (ONNX Shape Constraint Language)."""

from __future__ import annotations
from dataclasses import dataclass, field

__all__ = [
    "Expr",                # Base expression class
    "Statement",           # Base statement class
    "NumberLit",          # Integer literals
    "UnknownDim",         # ? literal
    "Identifier",         # Variable names
    "StringLit",          # String literals
    "BinOp",              # Binary operations (==, +, *, etc)
    "FuncCall",           # Function calls
    "ShapeLiteral",       # Shape tensors [...]
    "IndexExpr",          # Indexing (Xs[0])
    "IfExpr",             # if/then/else
    "MapExpr",            # map expressions
    "RequireStmt",        # require statements
    "LetStmt",            # let bindings
    "ResultStmt",         # result definitions
    "WhenStmt",           # when conditions
    "InputDecl",          # Input declarations
    "ShapeSpec",          # Complete shape spec
]

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
    # ... continues
```

**File size:** 158 lines total
**Key nodes:** Expr, Statement subclasses, all using Python dataclasses

---

## 11. PARSER & LEXER

### Lexer (oscl/lexer.py - 264 lines)
**Purpose:** Tokenizes OSCL source text

**Features:**
- 28 token types (keywords: shape, inputs, outputs, attributes, require, let, result, when, if, then, else, map, in, and, or)
- Operators: ==, !=, <, >, <=, >=, =, +, -, *, ?
- Delimiters: { } ( ) [ ] ; , :
- Literals: numbers, identifiers, strings
- **Context-aware negative number parsing** (recognized after operators, commas, parentheses)
- Comment support: `#` and `//`
- Line and column tracking for error reporting

### Parser (oscl/parser.py - 391 lines)
**Purpose:** Recursive-descent parser converting tokens to AST

**Architecture:**
- Single `_Parser` class with position tracking
- Recursive descent for expressions and statements
- Operator precedence handling (or < and < arithmetic < comparison)
- Error reporting with line/column information

**Main entry point:** `parse(source: str) -> ShapeSpec`

---

## 12. SPECIFICATION & DOCUMENTATION

### RFC Document (docs/rfc-oscl.md - 540 lines)

Complete formal specification covering:

| Topic | RFC Section |
|-------|-------------|
| Motivation | §1 |
| Goals / Non-Goals | §2, §3 |
| Type system (Rank/Dim/Shape) | §5 |
| Core syntax | §6 |
| Statements (require/let/result/when) | §7 |
| Dimension expressions | §8 |
| Shape operators | §9 |
| Shape tensor evaluation | §10 |
| Partial inference | §11 |
| Constraint predicates | §12 |
| Data-dependent operators | §13 |
| Example specifications | §14 |
| Canonical AST representation | §15 |
| Error semantics | §16 |
| ONNX integration | §17 |
| Reference implementation goals | §18 |
| Backward compatibility | §19 |

### agents.md - Implementation Guidance
Provides:
- Purpose statement and key concepts
- Quick reference to RFC sections
- Implementation guidelines (determinism, non-Turing-complete, partial inference)
- Suggested implementation structure
- Scope boundaries

---

## 13. MISSING: EVALUATOR/ENGINE CODE

**CRITICAL FINDING:** There is **NO evaluator, engine, or executor code** in this repository.

What exists:
- ✅ Lexer (tokenization)
- ✅ Parser (syntax → AST)
- ✅ AST definitions
- ✅ Spec file loader
- ✅ 13 example operator specifications

What does NOT exist:
- ❌ Evaluator/interpreter for executing shapes
- ❌ Shape inference engine
- ❌ Constraint validator
- ❌ Symbol table or environment
- ❌ Built-in function implementations (broadcast, dim, prefix, concat, etc.)
- ❌ Shape propagation logic
- ❌ ONNX integration

**Implication:** This is a parser/AST library only. To use OSCL, you would:
1. Parse specs using `parse()` or `load_spec()`
2. Get back a `ShapeSpec` AST
3. **Implement your own evaluator** to execute the shape rules

---

## 14. KEY IMPLEMENTATION DETAILS

### Code Statistics
```
oscl/__init__.py:    6 lines (API surface)
oscl/ast.py:       158 lines (AST definitions)
oscl/lexer.py:     264 lines (Tokenizer)
oscl/loader.py:     37 lines (Spec file loader)
oscl/parser.py:    391 lines (Parser)
────────────────────────────
Total:             856 lines
```

### Dependencies
- **Zero external dependencies** (only Python stdlib)
- Python 3.10+ required (uses modern syntax)

### Test Coverage
- **test_lexer.py:** 111 lines - Tokenization tests
- **test_parser.py:** 331 lines - Parsing & RFC example tests
- **test_loader.py:** 46 lines - Spec file loading tests
- **Parametrized test:** All 13 spec files are tested for successful parsing

### Design Patterns
- **Dataclasses:** All AST nodes are `@dataclass` instances
- **Recursive descent:** Parser uses hand-written recursive descent
- **Single entry point:** `parse(source: str) -> ShapeSpec`
- **No dependencies:** Everything in stdlib (dataclasses, enum, pathlib)

---

## 15. ARCHITECTURE SUMMARY

```
OSCL Reference Implementation

Input: OSCL Source Text
   ↓
[LEXER] → Tokens
   ↓
[PARSER] → ShapeSpec AST
   ↓
[USER CODE] → Shape Inference / Validation / Evaluation
              (NOT PROVIDED IN THIS REPO)
```

### What This Repo Provides
1. **Complete OSCL syntax parser** (lexer + parser)
2. **Full AST representation** matching RFC §15
3. **13 bundled operator specs** (matmul, concat, reshape, etc.)
4. **Spec loader utility** for runtime access
5. **Comprehensive test suite** with 100% spec file coverage

### What You Need to Provide (Not Included)
1. **Evaluator:** Execute shape expressions
2. **Built-in functions:** broadcast, dim, prefix, concat, etc.
3. **Symbol table:** Variable/attribute bindings
4. **Constraint checker:** Validate require statements
5. **ONNX integration:** Feed in model data, get shape inference

---

## 16. HOW TO USE THIS REPO

```python
from oscl import parse, load_spec, load_all_specs

# Parse from text
spec = parse("""
    shape {
        inputs A, B;
        outputs Y;
        require dim(A,-1) == dim(B,-2);
        result Y = [dim(A,-2), dim(B,-1)];
    }
""")

# Or load bundled specs
matmul_spec = load_spec("matmul")

# Or load all specs
all_specs = load_all_specs()
for op_name, spec in all_specs.items():
    print(f"{op_name}: {spec.outputs}")
```

The returned `ShapeSpec` object is a pure AST that you can introspect programmatically.

---

## 17. NEXT STEPS FOR MAKING CHANGES

Before any code changes:

1. **Read RFC** (`docs/rfc-oscl.md`) — It's the source of truth
2. **Check agents.md** — Implementation constraints and guidelines
3. **Review test_parser.py** — See how specifications are tested
4. **Understand scope** — This is parser/AST only, not an evaluator

Common modifications:
- **Add new spec files:** Add `.oscl` file to `oscl/specs/`
- **Change syntax:** Modify `lexer.py` and `parser.py`
- **Change AST:** Modify `ast.py`
- **Implement evaluator:** Would require new file(s) not yet created
- **Add tests:** Extend `tests/test_*.py`

---

