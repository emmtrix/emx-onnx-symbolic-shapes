# RFC: Declarative Type and Shape Rules for ONNX (OTSL)

## Status

Draft

## Authors

TBD

## Abstract

This document specifies a declarative domain-specific language (DSL) for ONNX type and shape inference called **ONNX Type and Shape Language (OTSL)**. OTSL is a normative representation of operator type and shape semantics. It is designed to be machine-readable, deterministic, and compatible with existing ONNX models, runtimes, and schema versioning rules.

The language core is intentionally small. OTSL contains only three statement forms:

- `require <predicate>`
- `let <name> = <expr>`
- `result <target> = <expr>`

Complex inference algorithms are expressed through **built-in functions**. Built-ins are normatively specified helper functions with fixed names, signatures, evaluation rules, and error conditions. This design keeps the DSL small, makes implementations easier to build, allows incremental migration from existing imperative ONNX inference code, and permits 100% of ONNX operators to be specified without introducing general computation into the language.

OTSL remains the human-readable authoring format. In this repository, the authoritative in-memory representation is the Python AST defined in `otsl.ast`.

---

## 1 Motivation

ONNX type and shape inference is currently defined primarily through imperative `TypeAndShapeInferenceFunction` implementations and helper routines such as `propagateShapeFromInputToOutput`. These implementations are useful, but they also have clear limitations:

- they are written as executable code rather than portable specifications
- they are difficult to analyze mechanically
- they are harder to validate for determinism and consistency
- they duplicate common logic across operators
- they make incremental tooling adoption more difficult

OTSL addresses these limitations by separating:

- a small declarative rule language used in operator schemas
- a normatively specified library of built-in helper functions used by those rules

This separation preserves the benefits of declarative specifications while accommodating the reality that many ONNX operators depend on non-trivial helper logic. Existing ONNX inference code can therefore be reused through built-ins instead of being translated immediately into a larger, more complex DSL.

The result is a specification system that:

- is deterministic
- supports symbolic dimension propagation
- supports partial inference
- remains compatible with current ONNX operator schemas
- permits incremental migration from imperative implementations
- allows all ONNX operators to be specified normatively

---

## 2 Goals

The proposed system must support:

1. symbolic dimension propagation
2. tensor element type propagation
3. constraints between dimensions and types
4. broadcast semantics
5. partial inference
6. attribute-driven inference behavior
7. reuse of existing ONNX helper logic through normative built-ins
8. specification coverage for 100% of ONNX operators
9. incremental migration from existing imperative inference functions

OTSL must remain:

- deterministic
- non-Turing-complete
- statically verifiable
- conservative when information is missing

---

## 3 Non-Goals

OTSL does not attempt to:

- describe full tensor computation semantics
- evaluate arbitrary runtime tensor values
- replace ONNX operator definitions or schema metadata
- introduce loops, recursion, or user-defined functions
- require immediate translation of every existing ONNX helper into primitive DSL syntax

---

## 4 Terminology

| Term | Meaning |
|------|---------|
| Rank | Number of tensor dimensions |
| Dim | Single tensor dimension term |
| Shape | Ordered list of dimensions |
| Type | Tensor element type |
| Predicate | Boolean constraint evaluated by `require` |
| Built-in | Normatively specified helper function used inside OTSL expressions |
| Shape-carrying input | Schema-designated integer tensor input whose abstract contents may be consumed by specific built-ins |

---

## 5 Type System

OTSL defines the following semantic domains.

### Rank

`rank(tensor)` is implemented as a non-negative integer derived from the known shape list of a tensor.

Example:

```
rank(A)
```

The current reference implementation does not carry a separate unknown-rank value. If a tensor shape is unavailable, expressions depending on rank usually cannot be evaluated and model-level inference degrades conservatively.

### Dim

A dimension term may be:

- a constant integer
- a direct input-derived dimension reference
- an expression over dimensions
- the unknown dimension marker `?`

Examples:

```
32
dim(A,0)
dim(A,0) + 5
?
```

OTSL uses the following dimension model.

| Kind | Notation | Meaning |
|------|----------|---------|
| known constant | `32` | exact non-negative runtime dimension |
| propagated dimension | `dim(A,0)` | symbolic dimension identified by an input or previously derived shape |
| expression | `dim(A,0) + 5` | deterministic term over dimensions |
| unknown | `?` | fresh unknown dimension term |

There is no `sym("N")` constructor. Symbolic relationships arise from repeated reuse of the same dimension term or `let` binding in a single evaluation.

The reference implementation also exposes `unknown_nonnegative()` as a built-in constructor for anonymous non-negative unknown dimensions. This is used by data-dependent operators such as `NonZero`, `Unique`, and `Compress`, where the result is unknown but should not introduce a fresh symbolic name into the emitted ONNX shape.

Each syntactic occurrence of `?` introduces a distinct fresh symbolic dimension variable at evaluation time.

Example:

```
[?, ?]
```

This shape introduces two independent dimension variables.

### Shape

A shape is an ordered list of dimensions.

Examples:

```
[]
[dim(A,0), dim(B,1)]
```

The relation between shape and rank is normative:

```
rank(T) = length(shape(T))
```

If `shape(T)` is known with length `k`, then `rank(T) = k`.

The current reference implementation does not synthesize an independent shape from a known rank. If no shape list is available for `T`, operations that depend on `shape(T)` or `rank(T)` usually cannot be evaluated and node-level inference degrades conservatively.

### Type

Type denotes a tensor element type.

Examples:

```
float
double
int64
type(A)
```

Type expressions may be compared and propagated.

Examples:

```
type(X)
type(X) == type(Y)
```

### Bool

Predicates are boolean-valued expressions used by `require` and conditional expressions.

Examples:

```
dim(A,0) == dim(B,0)
type(X) == type(Y)
```

### Type System Scope

OTSL directly models tensor shapes and tensor element types through `shape(...)`, `dim(...)`, `rank(...)`, and `type(...)`.

The reference implementation additionally supports full ONNX `TypeProto` values through the result field `output.onnx_type` and a small set of built-ins such as:

- `sequence_type(...)`
- `unwrap_optional_type(...)`
- `if_output_types()`
- `loop_output_types()`

Composite ONNX types are therefore not structurally authorable in the core expression grammar, but they are part of the implemented language surface via these built-ins and result fields.

---

## 6 Core Syntax

An OTSL rule block is embedded in an operator schema. Inputs, outputs, attributes, and variadic families are supplied by the surrounding ONNX schema. The DSL itself contains only three statement forms:

```
require <predicate>
let <name> = <expr>
result <target> = <expr>
```

Example:

```
rules {
  inputs Xs[];
  outputs Y;
  attributes axis;

  result Y.shape = concat_shape(Xs, axis);
  result Y.type = type(Xs[0]);
}
```

The core DSL does not include statement-level conditionals, shape-tensor declarations, or structural iteration constructs. These semantics are provided by expressions and built-ins.

---

## 7 Statements

### `require`

`require` declares a predicate that must hold for the operator invocation to be valid.

Examples:

```
require dim(A,1) == dim(B,0);
require type(X) == type(Y);
```

In the reference implementation, `require` is evaluated eagerly. A falsy result makes the current spec execution `invalid`.

### `let`

`let` binds an intermediate expression to a name.

Example:

```
let batch = broadcast(prefix(shape(A), 2), prefix(shape(B), 2));
```

`let` bindings are immutable and visible to later expressions in the same rule block.

### `result`

`result` assigns an expression to an output property.

Examples:

```
result Y.shape = [dim(A,0), dim(B,1)];
result Y.type = type(A);
```

The target must name an output field defined by the schema, such as `Y.shape` or `Y.type`.

Conditional behavior is expressed inside expressions rather than through a separate statement form:

```
result Y.shape = if rank(X) == 0 then [] else shape(X);
```

There is no `when` statement.

---

## 8 Expressions

OTSL expressions are first-order, side-effect-free terms. They may contain:

- literals
- input, output, attribute, and `let` references
- shape literals such as `[dim(A,0), 4]`
- arithmetic over dimensions
- comparisons and boolean combinations
- built-in function calls
- conditional expressions

Supported arithmetic over dimensions includes:

```
d1 + d2
d1 - d2
d1 * d2
floordiv(d1,d2)
```

Conditional behavior is expressed with:

```
if condition then expr1 else expr2
```

The reference implementation evaluates `if` eagerly using the runtime truthiness of the condition value. In practice this is primarily used with ONNX attributes such as `transA`, `transB`, `keepdims`, or similar `0`/`1` flags.

The DSL does not include:

- `when`
- `shape_tensors`
- `map`
- `range`
- `sum`
- set membership syntax such as `in { ... }`
- optional input declarations such as `B?`

Structural iteration must be expressed through built-ins. This keeps the language small and shifts complex, reusable algorithms into normatively specified helper functions.

Expressions that observe tensor shape and rank obey the following rules:

1. `rank(T)` is implemented as `len(shape(T))`.
2. `shape(T)` is available only when the reference engine has a concrete or partially known shape list for `T`.
3. If no shape is available for `T`, expression evaluation typically cannot proceed and model-level inference degrades conservatively by skipping inference for the current node.

The built-in `dim(tensor,index)` obeys the following rules:

1. Negative indices are normalized against the concrete known rank of the supplied shape.
2. If the normalized index is out of bounds, the call is `invalid`.
3. If the tensor shape is unavailable, evaluation typically fails for the current node and model-level inference degrades conservatively.

### Dimension Expression Canonicalization

The current reference implementation does not perform a separate global canonicalization or symbolic normalization pass. Expressions are evaluated eagerly to Python values, symbolic unknown names, ONNX `TypeProto` instances, or shape lists.

---

## 9 Built-in Functions

### Overview

The OTSL surface language is intentionally minimal. Most operator-specific behavior therefore lives in built-ins.

The current reference implementation ships a fixed built-in library in `otsl.numerical_engine` and `otsl.numpy_engine`. These built-ins fall into four rough groups:

- core shape helpers such as `shape`, `dim`, `rank`, `prefix`, `suffix`, `concat`, `broadcast`, and `normalize_axis`
- shape-input helpers such as `shape_value`, `resolve_reshape`, `slice_shape`, `resize_shape`, `split_shapes`, and `pad_shape`
- type and attribute helpers such as `type`, `attribute`, `input_type`, `attribute_value_type`, `attribute_value_shape`, and `attribute_values`
- composite-type helpers such as `sequence_type`, `sequence_elem_shape`, `sequence_elem_type`, `unwrap_optional_type`, `if_output_types`, and `loop_output_types`

Built-ins may be implemented either as pure argument-to-value functions or as environment-aware helpers that inspect known input types, ONNX attributes, or graph substructures.

### Current Contract

For the current reference implementation:

- built-in names are fixed by the dispatch tables in the engines
- a built-in either returns a runtime value or raises an exception
- exceptions are treated as spec execution failure for the current node
- model-level inference catches a bounded set of node-local failures and degrades conservatively

This RFC documents the current shipped language surface, not a separate symbolic calculus beyond what the implementation executes today.

### Operator-specific Built-ins

Many operators are intentionally specified as thin wrappers around reusable helper built-ins. Examples currently shipped in the reference implementation include:

- `conv_shape`
- `convtranspose_shape`
- `pool_shape`
- `global_pool_shape`
- `resize_shape`
- `einsum_shape`
- `rnn_shape`
- `stft_shape`
- `col2im_shape`
- `range_output_shape`

---

## 10 Built-in Library Governance

Built-ins are part of the ONNX specification.

Built-in names must be globally unique within the OTSL built-in library.

In a future ONNX integration, built-ins should be versioned together with the operator schema or opset in which they are used. The current repository ships one active implementation per built-in name.

Operator-specific built-ins must use descriptive names tied to operator semantics.

Examples include:

- `conv_output_shape`
- `matmul_output_shape`
- `slice_output_shape`

New built-ins require ONNX specification review before becoming normative.

---

## 11 Shape-carrying Inputs and Abstract Integer Sequences

Some ONNX operators accept tensor inputs whose statically available integer contents control type or shape inference. OTSL does not expose a dedicated `shape_tensors` statement. Instead, the surrounding operator schema identifies which inputs are shape-carrying for the built-ins that require them.

Examples include:

- the `shape` input of `Reshape`
- the `starts`, `ends`, `axes`, and `steps` inputs of `Slice` when statically available
- the scale or size inputs of `Resize` when statically available

Built-ins that accept a `ShapeInput` or related schema-designated input consume only the abstract integer information already available to ONNX inference. They do not evaluate arbitrary runtime tensor computation.

Available sources of such information include:

- constant initializers
- constant-folded shape carriers already available to inference
- schema-defined attribute-to-input conversions already visible to the inference engine

If the required contents are unavailable, the built-in must produce a conservative unresolved result instead of becoming implementation-defined or reading arbitrary runtime values.

---

## 12 Partial Inference

The reference implementation supports incomplete knowledge for both types and shapes, but does so with eager runtime values rather than a separate symbolic constraint solver.

Possible dimension forms include:

| Kind | Example |
|------|---------|
| known constant | `32` |
| symbolic unknown | `?` |
| anonymous unknown | `unknown_nonnegative()` |

The implementation distinguishes two common cases:

- `?` evaluates to a fresh symbolic dimension name such as `unk__0`
- operator-specific built-ins may return anonymous unknown dimensions represented without a symbolic name

Missing information is handled conservatively:

- a built-in may return a shape containing anonymous unknown dimensions
- a node may omit output inference for some fields
- model-level inference may skip a node entirely when the spec cannot be executed with the information currently available

The current implementation does not perform global equality propagation from `require` into earlier or later expressions.

---

## 13 Immediate Predicate Evaluation

The reference implementation does not implement a separate global constraint-solving phase.

Instead:

1. expressions are evaluated eagerly in source order
2. `require` checks the resulting predicate value immediately
3. falsy predicates raise `ConstraintViolation`
4. truthy predicates allow evaluation to continue

Successful `require` statements do not rewrite previously computed values, merge symbolic dimensions, or canonicalize expressions.

Built-in predicates such as `same_type`, `same_shape`, `compatible`, and `known` are not part of the currently shipped runtime unless explicitly implemented in the built-in dispatch tables.

---

## 14 Approximation for Data-dependent Operators

Some operators produce output shapes that depend on runtime tensor contents. OTSL must model these operators conservatively.

Example: `NonZero`.

```
rules {
  inputs X;
  outputs Y;

  result Y.shape = [rank(X), unknown_nonnegative()];
  result Y.type = int64;
}
```

This rule is valid because:

- the first output dimension is exactly the rank of `X`
- the second output dimension is data-dependent and therefore unknown
- the unknown dimension is still constrained to denote a non-negative extent

---

## 15 Example Specifications

### MatMul

```
rules {
  inputs A, B;
  outputs Y;

  require dim(A,-1) == dim(B,-2);
  let batch = broadcast(prefix(A,-2), prefix(B,-2));
  result Y.shape = concat(batch, [dim(A,-2), dim(B,-1)]);
  result Y.type = type(A);
}
```

The shipped `MatMul` spec is intentionally small and expresses the implemented batch-broadcasting rule directly in the DSL.

### Concat

```
rules {
  inputs Xs[];
  outputs Y;
  attributes axis;

  result Y.shape = concat_shape(Xs, axis);
  result Y.type = type(Xs[0]);
}
```

`concat_shape` replaces structural iteration such as `map`, `range`, and `sum`.

### Reshape

```
rules {
  inputs data, shape_input;
  outputs reshaped;
  attributes allowzero;

  let target = shape_value(shape_input);
  let az = attribute("allowzero", 0);
  result reshaped.shape = resolve_reshape(shape(data), target, az);
  result reshaped.type = type(data);
}
```

`resolve_reshape` consumes the schema-designated shape-carrying input `shape_input` through `shape_value(...)`.

### NonZero

```
rules {
  inputs X;
  outputs Y;

  result Y.shape = [rank(X), unknown_nonnegative()];
  result Y.type = int64;
}
```

This example has no loops, no structural iteration, and no dedicated shape-tensor syntax.

---

## 16 Reference AST

For this repository, the authoritative in-memory representation is the Python AST defined in `otsl.ast`.

The current AST includes:

- top-level declarations: `InputDecl` and `ShapeSpec`
- statement nodes: `RequireStmt`, `LetStmt`, `ResultStmt`
- expression nodes: `NumberLit`, `UnknownDim`, `Identifier`, `StringLit`, `BinOp`, `FuncCall`, `ShapeLiteral`, `IndexExpr`, `IfExpr`

The textual syntax is parsed by `otsl.parser` into this AST and then evaluated directly by the numerical and numpy engines.

Embedding an equivalent canonical AST into ONNX schema definitions is future work rather than current reference-implementation behavior.

---

## 17 Formal Evaluation Model

Evaluation of an OTSL rule block is deterministic and proceeds as follows:

1. Bind the schema-declared inputs, outputs, attributes, variadic families, and any schema metadata referenced by built-ins.
2. Initialize the environment with the available input shapes, element types, full ONNX types, attribute values, and any shape-carrying tensor values already available to inference.
3. Evaluate statements in source order.
4. For each `let`, evaluate the right-hand side eagerly and bind the resulting Python value to the given name.
5. For each `require`, evaluate the predicate eagerly. If the result is falsy, spec execution is `invalid`.
6. Successful `require` statements do not create a later normalization or equality-propagation phase.
7. For each `result`, evaluate the right-hand side in the current environment and assign it to the target output field.
8. Multiple assignments to the same output field are permitted only if the assigned runtime values are identical; otherwise spec execution is `invalid`.
9. Built-in calls are evaluated according to their normative specifications. A built-in may:
   - return an exact value
   - return a symbolic unknown name or anonymous unknown dimension
   - determine that the invocation is `invalid` under its specified error conditions
10. There is no final normalization pass after all statements have been processed.
11. In the public graph inference API, node-local execution failures such as `ConstraintViolation`, `ValueError`, `TypeError`, `IndexError`, `KeyError`, `NameError`, `ZeroDivisionError`, or `AttributeError` are caught and treated as conservative degradation: inference for that node is skipped rather than aborting the whole model.

This algorithm defines the observable behavior of the current reference implementation.

---

## 18 Error Semantics

The current reference implementation distinguishes two layers:

- spec execution (`_execute_spec`) may either produce output field values or raise an exception such as `ConstraintViolation` or `ValueError`
- model-level inference (`infer_shapes`) catches a bounded set of node-local evaluation exceptions and degrades conservatively by leaving that node less inferred

The public API does not currently expose a separate `exact` / `symbolic` / `partial` / `unknown` status lattice.

---

## 19 Integration with ONNX

A future ONNX integration may attach OTSL rule specifications to `OpSchema` definitions.

In this repository, OTSL rules are consumed directly by the Python reference engines rather than embedded into ONNX schema metadata.

In this repository, specs are currently loaded from `otsl/specs/*.otsl` and keyed by lowercase `op_type`. Domain-specific or opset-specific dispatch is future work.

If both an OTSL rule set and an imperative `TypeAndShapeInferenceFunction` are present for the same schema version, they must be semantically equivalent. The OTSL rule is the normative portable specification. The imperative function is an allowed implementation technique and backward-compatibility mechanism.

A runtime that does not interpret OTSL may:

- execute an existing imperative inference implementation
- use generated code derived from OTSL and built-ins
- perform no inference at all

Lack of OTSL support must not change ONNX model validity. It changes only the amount of inference the runtime can perform.

---

## 20 Migration from Imperative Inference Functions

Existing `TypeAndShapeInferenceFunction` implementations may be adopted incrementally by wrapping them as built-ins.

This migration strategy is normative and intentional:

- the DSL stays small
- existing inference code remains reusable
- operators can be specified incrementally instead of requiring a complete rewrite
- behavior remains aligned with current ONNX semantics during transition

Example migration:

Old ONNX schema implementation:

```cpp
TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
  // existing convolution inference logic
});
```

Refactored OTSL rule:

```
rules {
  inputs X, W;
  outputs Y;
  attributes auto_pad, dilations, group, kernel_shape, pads, strides;

  result Y.shape =
    conv_shape(shape(X), shape(W), kernel_shape, strides, pads, dilations, group, auto_pad);
  result Y.type = type(X);
}
```

`conv_shape` may be implemented by reusing existing imperative convolution inference logic, provided the built-in is documented with a fixed signature, deterministic evaluation rules, and explicit error conditions.

This approach allows operator coverage to grow immediately while leaving room for later decomposition of a built-in into smaller shared helpers when that is beneficial.

---

## 21 Reference Implementation Goals

A reference implementation should:

- parse the OTSL textual format
- represent the reference AST used by the implementation
- evaluate the three core statement forms
- implement the shipped built-in library
- propagate symbolic shapes and types
- support conservative partial inference
- validate eager predicate checks deterministically

Suitable implementation languages include:

- Python
- C++
- Rust

---

## 22 Backward Compatibility

This proposal does not modify ONNX graph semantics.

Existing ONNX models remain valid.

Existing runtimes may:

- ignore OTSL rules
- interpret OTSL directly
- lower OTSL to generated code
- continue using imperative inference implementations

Incremental migration through built-ins is explicitly compatible with current ONNX deployment practice.

---

## 23 Future Work

Possible future extensions include:

- richer built-in libraries shared across operator families
- interval reasoning for dimensions
- stronger automatic validation of built-in specifications
- mechanical equivalence checking between OTSL rules and legacy inference code
- integration with graph optimization tooling

Any future extension must preserve the determinism and minimal core syntax defined by this RFC.

---

## 24 Conclusion

OTSL defines a normative, declarative, machine-readable specification for ONNX type and shape inference.

By reducing the DSL to three statement forms and moving complex semantics into normatively specified built-ins, this RFC keeps the language small, enables incremental migration from existing ONNX inference code, and makes full operator coverage practical without sacrificing determinism or compatibility.
