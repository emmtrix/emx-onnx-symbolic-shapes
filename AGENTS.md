# AGENTS.md - AI Agent Guidance

This file provides context and instructions for AI coding agents working in this repository.

## Purpose

This repository is the **reference implementation** for OTSL (ONNX Type and Shape Language), a declarative DSL for specifying shape inference rules for ONNX operators.

The normative specification is the RFC located at [`docs/rfc-otsl.md`](docs/rfc-otsl.md).
All implementation decisions must be grounded in the RFC. When the RFC is ambiguous or silent on a topic, prefer the most conservative, deterministic interpretation.

## Key Concepts (summary from RFC)

- **OTSL** defines shape rules for ONNX operators using symbolic expressions and constraints.
- Rules are written in a declarative, non-Turing-complete DSL embedded inside operator definitions.
- The canonical representation is a structured AST (see RFC section 15); the textual syntax is a presentation format.
- Shape dimensions may be: known constants, symbolic variables, expressions, or unknown (`?`).
- Inference is partial: a rule may produce `symbolic`, `partial`, or `unknown` results - never an error for missing information.
- Constraint violations (`invalid` status) indicate an ill-formed graph.

## RFC Sections Quick Reference

| Topic | RFC Section |
|-------|-------------|
| Motivation | section 1 |
| Goals / Non-Goals | sections 2, 3 |
| Type system (Rank/Dim/Shape) | section 5 |
| Core syntax | section 6 |
| Statements (require/let/result/when) | section 7 |
| Dimension expressions | section 8 |
| Shape operators | section 9 |
| Shape tensor evaluation | section 10 |
| Partial inference | section 11 |
| Constraint predicates | section 12 |
| Data-dependent operators | section 13 |
| Example specifications | section 14 |
| Canonical AST representation | section 15 |
| Error semantics | section 16 |
| ONNX integration | section 17 |
| Reference implementation goals | section 18 |
| Backward compatibility | section 19 |

## Implementation Guidelines

1. **RFC is the source of truth.** Before implementing any shape rule, read the corresponding RFC section.
2. **Determinism is mandatory.** Shape inference must produce the same result for identical inputs.
3. **Non-Turing-complete.** Do not introduce loops, recursion, or general computation into OTSL evaluation.
4. **Partial inference is a first-class feature.** Never fail hard when a dimension is unknown; propagate `?` instead.
5. **Constraints vs. shapes.** `require` statements assert pre-conditions; `result` statements define outputs. Keep them separate.
6. **AST representation.** New operators should be representable in the JSON AST format defined in RFC section 15.
7. **Backward compatibility.** Changes must not invalidate existing ONNX models (RFC section 19).
8. **No operator-specific engine semantics.** Do not implement ONNX operator behavior via `if node.op_type == ...` or similar special cases in the engine, evaluator, parser, or tests. Operator semantics must live in OTSL specs or in generic, operator-agnostic language/runtime primitives. The only acceptable use of `op_type` is spec lookup, diagnostics, or other purely generic dispatch that does not encode operator-specific inference logic.

## Suggested Implementation Structure

```
otsl/
  parser/       # textual DSL -> AST
  ast/          # AST node definitions
  evaluator/    # symbolic shape propagation
  validator/    # constraint checking
  examples/     # per-operator OTSL definitions (MatMul, Concat, Reshape, ...)
tests/
  test_parser.py
  test_evaluator.py
  test_validator.py
docs/
  rfc-otsl.md   # normative RFC
```

## Out of Scope

The following are explicitly **not** in scope (RFC section 3):

- Full tensor computation semantics
- Arbitrary tensor value evaluation
- Replacing ONNX operator definitions
- General programming constructs inside shape rules
