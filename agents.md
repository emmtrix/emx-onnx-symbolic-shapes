# agents.md – AI Agent Guidance

This file provides context and instructions for AI coding agents working in this repository.

## Purpose

This repository is the **reference implementation** for OSCL (ONNX Shape Constraint Language), a declarative DSL for specifying shape inference rules for ONNX operators.

The normative specification is the RFC located at [`docs/rfc-oscl.md`](docs/rfc-oscl.md).  
All implementation decisions must be grounded in the RFC. When the RFC is ambiguous or silent on a topic, prefer the most conservative, deterministic interpretation.

## Key Concepts (summary from RFC)

- **OSCL** defines shape rules for ONNX operators using symbolic expressions and constraints.
- Rules are written in a declarative, non-Turing-complete DSL embedded inside operator definitions.
- The canonical representation is a structured AST (see RFC §15); the textual syntax is a presentation format.
- Shape dimensions may be: known constants, symbolic variables, expressions, or unknown (`?`).
- Inference is partial: a rule may produce `symbolic`, `partial`, or `unknown` results – never an error for missing information.
- Constraint violations (`invalid` status) indicate an ill-formed graph.

## RFC Sections Quick Reference

| Topic                        | RFC Section |
|------------------------------|-------------|
| Motivation                   | §1          |
| Goals / Non-Goals            | §2, §3      |
| Type system (Rank/Dim/Shape) | §5          |
| Core syntax                  | §6          |
| Statements (require/let/result/when) | §7  |
| Dimension expressions        | §8          |
| Shape operators              | §9          |
| Shape tensor evaluation      | §10         |
| Partial inference            | §11         |
| Constraint predicates        | §12         |
| Data-dependent operators     | §13         |
| Example specifications       | §14         |
| Canonical AST representation | §15         |
| Error semantics              | §16         |
| ONNX integration             | §17         |
| Reference implementation goals | §18       |
| Backward compatibility       | §19         |

## Implementation Guidelines

1. **RFC is the source of truth.** Before implementing any shape rule, read the corresponding RFC section.
2. **Determinism is mandatory.** Shape inference must produce the same result for identical inputs.
3. **Non-Turing-complete.** Do not introduce loops, recursion, or general computation into OSCL evaluation.
4. **Partial inference is a first-class feature.** Never fail hard when a dimension is unknown; propagate `?` instead.
5. **Constraints vs. shapes.** `require` statements assert pre-conditions; `result` statements define outputs. Keep them separate.
6. **AST representation.** New operators should be representable in the JSON AST format defined in RFC §15.
7. **Backward compatibility.** Changes must not invalidate existing ONNX models (RFC §19).

## Suggested Implementation Structure

```
oscl/
  parser/       # textual DSL → AST
  ast/          # AST node definitions
  evaluator/    # symbolic shape propagation
  validator/    # constraint checking
  examples/     # per-operator OSCL definitions (MatMul, Concat, Reshape, …)
tests/
  test_parser.py
  test_evaluator.py
  test_validator.py
docs/
  rfc-oscl.md   # normative RFC
```

## Out of Scope

The following are explicitly **not** in scope (RFC §3):

- Full tensor computation semantics
- Arbitrary tensor value evaluation
- Replacing ONNX operator definitions
- General programming constructs inside shape rules
