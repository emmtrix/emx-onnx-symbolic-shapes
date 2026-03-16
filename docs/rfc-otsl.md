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

OTSL remains the human-readable authoring format. The canonical representation is a structured AST embedded in the ONNX `OpSchema` definition.

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

Rank is a non-negative integer.

Example:

```
rank(A)
```

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

There is no `sym("N")` constructor. Symbolic relationships arise from repeated use of the same dimension term and from equality constraints. For example, `dim(A,0)` denotes the same symbolic dimension wherever it appears, and `require dim(A,0) == dim(B,1)` unifies those dimensions for later evaluation.

There is no `unknown_nonnegative()` constructor. All dimension terms are implicitly constrained to be non-negative because they denote tensor extents.

Each evaluation of `?` produces a fresh unknown dimension term. Two syntactic occurrences of `?` are equal only if later constraints prove them equal.

### Shape

A shape is an ordered list of dimensions.

Examples:

```
[]
[dim(A,0), dim(B,1)]
```

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
type(X) in {float, double}
```

### Bool

Predicates are boolean-valued expressions used by `require` and conditional expressions.

Examples:

```
dim(A,0) == dim(B,0)
type(X) == type(Y)
known(dim(A,0))
```

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

  require same_type(Xs);
  let ax = normalize_axis(axis, rank(Xs[0]));
  result Y.shape = concat_shape(Xs, ax);
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
require type(X) in {float, double};
```

If a `require` predicate is proved false, the rule result is `invalid`.

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
- comparisons and set membership
- built-in function calls
- conditional expressions

Supported arithmetic over dimensions includes:

```
d1 + d2
d1 - d2
d1 * d2
floordiv(d1,d2)
ceildiv(d1,d2)
max(d1,d2)
min(d1,d2)
```

Conditional behavior is expressed with:

```
if condition then expr1 else expr2
```

Conditional expressions use three-valued predicate evaluation:

- if `condition` is `true`, the result is `expr1`
- if `condition` is `false`, the result is `expr2`
- if `condition` is `unknown`, the conditional expression is unresolved unless both branches normalize to the same value

The DSL does not include:

- `when`
- `shape_tensors`
- `map`
- `range`
- `sum`

Structural iteration must be expressed through built-ins. This keeps the language small and shifts complex, reusable algorithms into normatively specified helper functions.

Dimension expressions are normalized only by deterministic local simplifications such as constant folding, substitution of solved equalities, and elimination of identity operations. General algebraic search or rearrangement is not required.

---

## 9 Built-in Functions

### Overview

The OTSL surface language is intentionally minimal. Complex inference algorithms are provided through built-in functions.

Built-ins are:

- deterministic
- pure
- side-effect-free
- normatively specified
- identified by stable names and signatures

Built-ins may correspond directly to existing ONNX helper logic or existing `TypeAndShapeInferenceFunction` implementations, provided the semantics remain the normative semantics defined by this RFC.

Built-ins are part of the OTSL specification. They are not implementation-defined extension points.

### Common Built-ins

Common built-ins include:

| Name | Signature | Meaning |
|------|-----------|---------|
| `shape` | `shape(tensor: Tensor) -> Shape` | Returns the tensor shape when available |
| `dim` | `dim(tensor: Tensor, index: Int) -> Dim` | Returns one dimension, supporting negative indices |
| `prefix` | `prefix(value: Shape, count: Int) -> Shape` | Returns the leading dimensions |
| `suffix` | `suffix(value: Shape, count: Int) -> Shape` | Returns the trailing dimensions |
| `concat` | `concat(left: Shape, right: Shape) -> Shape` | Concatenates two shapes |
| `permute` | `permute(value: Shape, order: IntSeq) -> Shape` | Applies a permutation |
| `broadcast` | `broadcast(left: Shape, right: Shape) -> Shape` | Computes ONNX broadcast shape |
| `normalize_axis` | `normalize_axis(axis: Int, rank: Rank) -> Int` | Normalizes a possibly negative axis |
| `resolve_reshape` | `resolve_reshape(input_shape: Shape, target: ShapeInput, allowzero: Int) -> Shape` | Computes ONNX reshape result |
| `concat_shape` | `concat_shape(inputs: TensorSeq, axis: Int) -> Shape` | Computes ONNX concat result shape |

### Built-in Specification Format

Every built-in specification in this RFC or in future extensions must contain the following fields:

- `Name`
- `Signature`
- `Input domain`
- `Evaluation rules`
- `Error conditions`

The signature defines the arity and abstract domains of the arguments and result. The input domain defines when the built-in is applicable. The evaluation rules define the exact deterministic result. The error conditions define when the built-in causes inference to return `invalid` instead of a partial or unknown result.

Auxiliary signature domains used in built-in specifications are:

- `Tensor`: a schema-declared tensor input or output
- `TensorSeq`: a schema-declared variadic tensor input family
- `ShapeInput`: a schema-declared shape-carrying input
- `IntSeq`: a finite abstract integer sequence whose entries may be known integers or unknown entries

### Example Built-in Specifications

#### `broadcast`

- `Name`: `broadcast`
- `Signature`: `broadcast(left: Shape, right: Shape) -> Shape`
- `Input domain`:
  - `left` and `right` are finite shapes.
  - Each dimension is a non-negative dimension term.
- `Evaluation rules`:
  1. Let `m = len(left)`, `n = len(right)`, and `r = max(m, n)`.
  2. Align `left` and `right` by trailing dimension. Missing leading dimensions are treated as constant `1`.
  3. For each aligned pair `(a, b)`:
     - if `normalize(a)` and `normalize(b)` are syntactically identical, the result dimension is that normalized term
     - else if `normalize(a) == 1`, the result dimension is `normalize(b)`
     - else if `normalize(b) == 1`, the result dimension is `normalize(a)`
     - else if both normalize to distinct known constants greater than `1`, inference is `invalid`
     - else the result dimension is `?`
  4. The result shape is the ordered list of the dimensions produced in step 3.
- `Error conditions`:
  - any aligned pair of dimensions that are provably incompatible by ONNX broadcast rules makes the call `invalid`
  - unavailable input information does not itself make the call `invalid`; it yields `?` in the corresponding output dimension

#### `normalize_axis`

- `Name`: `normalize_axis`
- `Signature`: `normalize_axis(axis: Int, rank: Rank) -> Int`
- `Input domain`:
  - `rank` denotes the rank of the tensor to which the axis applies.
  - `axis` is an integer-valued attribute or integer expression available to inference.
- `Evaluation rules`:
  1. If `rank` is a known constant `r` and `r == 0`, inference is `invalid`.
  2. If `rank` is a known constant `r` and `axis` is a known constant `a`:
     - if `0 <= a < r`, return `a`
     - else if `-r <= a < 0`, return `a + r`
     - else inference is `invalid`
  3. If either argument is not known exactly, the result is unresolved.
- `Error conditions`:
  - rank `0`
  - a known axis outside the closed-open interval `[-rank, rank)`

#### `resolve_reshape`

- `Name`: `resolve_reshape`
- `Signature`: `resolve_reshape(input_shape: Shape, target: ShapeInput, allowzero: Int) -> Shape`
- `Input domain`:
  - `input_shape` is the input tensor shape.
  - `target` is the schema-designated shape-carrying input of `Reshape`.
  - `allowzero` is either `0` or `1`.
- `Evaluation rules`:
  1. Read `target` as an abstract integer sequence `t`.
  2. If the length of `t` is unavailable, the result is unresolved.
  3. For each index `i` in `t`:
     - if `t[i] > 0`, output dimension `i` is the constant `t[i]`
     - if `t[i] == 0` and `allowzero == 0`, output dimension `i` is `input_shape[i]`
     - if `t[i] == 0` and `allowzero == 1`, output dimension `i` is `0`
     - if `t[i] == -1`, mark index `i` as the inferred dimension position
     - if `t[i]` is unknown, output dimension `i` is `?`
     - if `t[i] < -1`, inference is `invalid`
  4. At most one index may be marked by `-1`. More than one such index is `invalid`.
  5. If no `-1` occurs:
     - if both the total element count of `input_shape` and the total element count of the candidate output shape are known and unequal, inference is `invalid`
     - otherwise return the candidate output shape
  6. If exactly one `-1` occurs:
     - let `known_product` be the product of all output dimensions other than the `-1` position after applying the rules above
     - if `input_shape` has a known total element count and `known_product` is a known positive integer that divides that count evenly, replace `-1` with the quotient
     - if element counts are known and no valid quotient exists, inference is `invalid`
     - otherwise replace the `-1` position with `?`
- `Error conditions`:
  - `allowzero` not in `{0,1}`
  - any target entry smaller than `-1`
  - more than one `-1`
  - copying with `0` when `allowzero == 0` and the corresponding input dimension does not exist
  - known input and output element counts that contradict ONNX `Reshape`

#### `concat_shape`

- `Name`: `concat_shape`
- `Signature`: `concat_shape(inputs: TensorSeq, axis: Int) -> Shape`
- `Input domain`:
  - `inputs` is a non-empty variadic tensor family
  - `axis` is an integer axis for those tensors
- `Evaluation rules`:
  1. If `inputs` is empty, inference is `invalid`.
  2. Let `r = rank(inputs[0])`.
  3. If `r` is unavailable, the result is unresolved.
  4. If any input has a known rank different from `r`, inference is `invalid`.
  5. Let `ax = normalize_axis(axis, r)`.
  6. For each dimension position `j` with `0 <= j < r`:
     - if `j != ax`, compare `dim(inputs[k], j)` for all `k`
       - if any pair is provably unequal, inference is `invalid`
       - otherwise the result dimension is `normalize(dim(inputs[0], j))`
     - if `j == ax`, the result dimension is the left-associated sum of `normalize(dim(inputs[k], j))` for all `k`, with constant folding applied when possible; if any participating term is completely unavailable, the result dimension is `?`
  7. The result shape is the ordered list produced in step 6.
- `Error conditions`:
  - empty input family
  - provably inconsistent non-axis dimensions
  - known ranks that disagree
  - invalid axis

### Operator-specific Built-ins

Some operators are most naturally expressed through dedicated helper built-ins rather than primitive expression combinations. This is permitted and expected.

Examples include:

- `conv_output_shape`
- `slice_output_shape`
- `resize_output_shape`
- `gather_output_shape`
- `matmul_output_shape`

These built-ins correspond to normatively specified ONNX inference behavior and may be implemented by reusing existing ONNX helper logic. Their purpose is to keep the DSL small while still allowing the full operator set to be specified.

---

## 10 Shape-carrying Inputs and Abstract Integer Sequences

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

## 11 Partial Inference

OTSL supports incomplete knowledge for both types and shapes.

Possible dimension forms include:

| Kind | Example |
|------|---------|
| known constant | `32` |
| propagated dimension | `dim(A,0)` |
| expression | `dim(A,0) + 5` |
| unknown | `?` |

Partial inference is fundamental. Missing information must not cause hard failure unless a rule proves the operator invocation invalid.

Examples:

```
require dim(A,0) == dim(B,0)
result Y.shape = [dim(A,0), ?]
```

Equality constraints may refine unknown or symbolic relationships even when exact values are not known.

Type inference may likewise remain unresolved when the rule block does not determine a unique element type.

---

## 12 Constraint Solving and Built-in Constraint Predicates

OTSL constraint solving is defined over normalized dimension and type terms. Let `normalize(t)` recursively substitute solved equalities into `t` and apply deterministic local simplifications such as constant folding.

Equality constraints on dimensions are processed using the following rules:

1. `constant == constant` succeeds iff the constants are identical; otherwise inference is `invalid`.
2. Repeated occurrences of the same normalized dimension term denote the same symbolic entity.
3. `dim(...) == dim(...)` merges the participating representatives into one equivalence class.
4. `? == term` binds that fresh unknown term to `normalize(term)`, provided this does not create a cycle.
5. `representative == expression` binds the representative to the normalized expression iff the representative does not occur within that expression.
6. `expression == expression` succeeds when the normalized expressions are syntactically identical. If both normalize to different constants, inference is `invalid`. Otherwise the equality remains unresolved.

Cyclic bindings are contradictions and therefore `invalid`.

Every occurrence of `?` is fresh. If later equalities unify that fresh term with a constant, direct input dimension, or expression, that information propagates to every use of the same term created by that evaluation.

Because dimensions denote tensor extents, any solved equality that forces a dimension to a negative known constant is a contradiction and therefore `invalid`.

Constraint propagation is monotonic and deterministic. Implementations may refine terms by:

- substitution of solved equalities
- equivalence-class merging
- constant folding
- local deterministic simplifications

Implementations must not rely on:

- backtracking
- implementation-defined search
- non-deterministic rewrite ordering
- general-purpose symbolic algebra

A `require` predicate evaluates in three-valued logic after normalization:

- `true`: the predicate is satisfied
- `false`: the rule result is `invalid`
- `unknown`: the predicate remains unresolved and the overall result cannot be more precise than `partial`

Common built-in constraint predicates include:

```
compatible(d1,d2)
same_shape(A,B)
same_type(Xs)
known(d)
```

These predicates are defined as follows:

- `known(d)` is `true` iff `normalize(d)` is a constant
- `compatible(d1,d2)` is `false` only when the two normalized dimensions are provably unequal, `true` when they are provably equal, and `unknown` otherwise
- `same_shape(A,B)` compares shapes pointwise using rank equality and `compatible`
- `same_type(Xs)` is `true` iff all known element types in the family are identical, `false` if any two are provably different, and `unknown` otherwise

Type equality and set membership are exact. No subtype relation is introduced.

---

## 13 Approximation for Data-dependent Operators

Some operators produce output shapes that depend on runtime tensor contents. OTSL must model these operators conservatively.

Example: `NonZero`.

```
rules {
  inputs X;
  outputs Y;

  result Y.shape = [rank(X), ?];
  result Y.type = int64;
}
```

This rule is valid because:

- the first output dimension is exactly the rank of `X`
- the second output dimension is data-dependent and therefore unknown
- the unknown dimension is still implicitly non-negative because all dimensions are non-negative

---

## 14 Example Specifications

### MatMul

```
rules {
  inputs A, B;
  outputs Y;

  require type(A) == type(B);
  result Y.shape = matmul_output_shape(A, B);
  result Y.type = type(A);
}
```

`matmul_output_shape` is an operator-specific built-in implementing the normative ONNX `MatMul` shape rules, including 1-D promotion, batch broadcasting, and output rank reduction.

### Concat

```
rules {
  inputs Xs[];
  outputs Y;
  attributes axis;

  require same_type(Xs);
  result Y.shape = concat_shape(Xs, axis);
  result Y.type = type(Xs[0]);
}
```

`concat_shape` replaces structural iteration such as `map`, `range`, and `sum`.

### Reshape

```
rules {
  inputs data, shape;
  outputs reshaped;
  attributes allowzero;

  result reshaped.shape = resolve_reshape(shape(data), shape, allowzero);
  result reshaped.type = type(data);
}
```

`resolve_reshape` consumes the schema-designated shape-carrying input `shape` and applies the exact ONNX `Reshape` rules.

### NonZero

```
rules {
  inputs X;
  outputs Y;

  result Y.shape = [rank(X), ?];
  result Y.type = int64;
}
```

This example has no loops, no structural iteration, and no dedicated shape-tensor syntax.

---

## 15 Canonical Representation

The normative representation of OTSL is a structured AST. The textual syntax is a presentation format for authoring, review, and generated documentation.

The AST contains:

- schema references for inputs, outputs, attributes, and variadic families
- statement nodes of kind `require`, `let`, and `result`
- expression nodes, including built-in calls and conditional expressions
- schema metadata required by built-ins, such as shape-carrying input designations

Example (simplified JSON):

```json
{
  "statements": [
    {
      "kind": "require",
      "expr": {
        "op": "eq",
        "args": [
          { "op": "type", "tensor": "A" },
          { "op": "type", "tensor": "B" }
        ]
      }
    },
    {
      "kind": "result",
      "target": { "tensor": "Y", "field": "shape" },
      "expr": {
        "op": "builtin",
        "name": "matmul_output_shape",
        "args": [
          { "op": "input", "name": "A" },
          { "op": "input", "name": "B" }
        ]
      }
    },
    {
      "kind": "result",
      "target": { "tensor": "Y", "field": "type" },
      "expr": { "op": "type", "tensor": "A" }
    }
  ]
}
```

### C++ Shape and Type Rule Representation

OTSL text is only a presentation format. The canonical representation used by ONNX schemas is an AST embedded in the C++ `OpSchema` definition.

Example:

```cpp
OpSchema()
  .SetTypeAndShapeRules(
      Require(
          Eq(TypeOf("A"), TypeOf("B"))
      ),
      Result(
          OutputField("Y", "shape"),
          Builtin("matmul_output_shape", Input("A"), Input("B"))
      ),
      Result(
          OutputField("Y", "type"),
          TypeOf("A")
      )
  );
```

The AST must encode only the minimal statement set plus the built-in calls and metadata needed to interpret them.

---

## 16 Formal Evaluation Model

Evaluation of an OTSL rule block is deterministic and proceeds as follows:

1. Bind the schema-declared inputs, outputs, attributes, variadic families, and any schema metadata referenced by built-ins.
2. Initialize the environment with the available input ranks, shapes, element types, attribute values, and abstract integer information already available to inference.
3. Evaluate statements in source order.
4. For each `let`, evaluate the right-hand side, normalize it with the current solved equalities, and bind the resulting expression to the given name.
5. For each `require`, evaluate and normalize the predicate, add any equality information to the current constraint set, and classify the predicate as `true`, `false`, or `unknown`.
6. If any active `require` becomes `false`, inference is `invalid`.
7. For each `result`, evaluate the right-hand side in the current environment and assign it to the target output field.
8. Multiple assignments to the same output field are permitted only if the normalized assigned expressions are identical; otherwise inference is `invalid`.
9. Built-in calls are evaluated according to their normative specifications. A built-in may:
   - return an exact value
   - return a symbolic or partial value containing direct dimension terms, expressions, or `?`
   - determine that the invocation is `invalid` under its specified error conditions
10. After all statements have been processed, normalize every output expression with the final solved equalities.
11. Classify the final result:
    - `invalid` if any contradiction or built-in error condition was derived
    - `exact` if every output type and dimension is fully known
    - `symbolic` if outputs are determined and contain expressions over known symbolic dimension terms but no `?`
    - `partial` if some outputs are derived but contain `?` or unresolved predicates
    - `unknown` if no output property can be derived

This algorithm defines the observable semantics. Implementations may optimize it, but any equivalent implementation must produce the same normalized outputs and status for the same inputs.

---

## 17 Error Semantics

Evaluation of OTSL rules may produce:

| Status | Meaning |
|--------|---------|
| `exact` | type and shape fully known |
| `symbolic` | output uses symbolic dimension terms or expressions |
| `partial` | output derived but some information remains unknown |
| `unknown` | no output property can be derived |
| `invalid` | the operator invocation violates required constraints |

Missing information is not an error. Proven contradiction is an error.

---

## 18 Integration with ONNX

Each `OpSchema` may include an OTSL rule specification.

OTSL rules are versioned with the ONNX operator schema or opset version to which they apply. Built-ins are likewise versioned by the schema context in which they are interpreted. A semantic change to a rule or a built-in requires the same schema versioning discipline already used by ONNX.

If both an OTSL rule set and an imperative `TypeAndShapeInferenceFunction` are present for the same schema version, they must be semantically equivalent. The OTSL rule is the normative portable specification. The imperative function is an allowed implementation technique and backward-compatibility mechanism.

A runtime that does not interpret OTSL may:

- execute an existing imperative inference implementation
- use generated code derived from OTSL and built-ins
- perform no inference at all

Lack of OTSL support must not change ONNX model validity. It changes only the amount of inference the runtime can perform.

---

## 19 Migration from Imperative Inference Functions

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
  inputs X, W, B?;
  outputs Y;
  attributes auto_pad, dilations, group, kernel_shape, pads, strides;

  result Y.shape =
    conv_output_shape(X, W, auto_pad, dilations, group, kernel_shape, pads, strides);
  result Y.type = type(X);
}
```

`conv_output_shape` may initially be implemented by reusing the existing imperative convolution inference logic, provided the built-in is documented with a fixed signature, deterministic evaluation rules, and explicit error conditions.

This approach allows operator coverage to grow immediately while leaving room for later decomposition of a built-in into smaller shared helpers when that is beneficial.

---

## 20 Reference Implementation Goals

A reference implementation should:

- parse the OTSL textual format
- represent the canonical AST
- evaluate the three core statement forms
- implement the normative built-in library
- propagate symbolic shapes and types
- support partial inference
- validate constraints deterministically

Suitable implementation languages include:

- Python
- C++
- Rust

---

## 21 Backward Compatibility

This proposal does not modify ONNX graph semantics.

Existing ONNX models remain valid.

Existing runtimes may:

- ignore OTSL rules
- interpret OTSL directly
- lower OTSL to generated code
- continue using imperative inference implementations

Incremental migration through built-ins is explicitly compatible with current ONNX deployment practice.

---

## 22 Future Work

Possible future extensions include:

- richer built-in libraries shared across operator families
- interval reasoning for dimensions
- stronger automatic validation of built-in specifications
- mechanical equivalence checking between OTSL rules and legacy inference code
- integration with graph optimization tooling

Any future extension must preserve the determinism and minimal core syntax defined by this RFC.

---

## 23 Conclusion

OTSL defines a normative, declarative, machine-readable specification for ONNX type and shape inference.

By reducing the DSL to three statement forms and moving complex semantics into normatively specified built-ins, this RFC keeps the language small, enables incremental migration from existing ONNX inference code, and makes full operator coverage practical without sacrificing determinism or compatibility.
