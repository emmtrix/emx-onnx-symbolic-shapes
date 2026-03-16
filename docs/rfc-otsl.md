# RFC: Declarative Type and Shape Rules for ONNX (OTSL)

## Status

Draft

## Authors

TBD

## Abstract

This document proposes a declarative domain-specific language (DSL) for specifying type and shape inference rules for ONNX operators. The language, called **ONNX Type and Shape Language (OTSL)**, allows operator schemas to define input/output type and shape relationships using symbolic expressions and constraints.

The goal is to replace or complement existing imperative type and shape inference implementations with a machine-readable, declarative specification that:

- is deterministic
- supports symbolic dimensions
- supports tensor element type propagation
- supports partial inference
- is suitable for static analysis
- can be validated automatically

OTSL remains the human-readable specification language, while the canonical representation is an AST embedded in the ONNX C++ `OpSchema` definition. OTSL is intended to become a normative representation of operator type and shape semantics while remaining compatible with existing ONNX models and runtimes.

---

## 1 Motivation

Type and shape inference in ONNX is currently implemented as imperative functions attached to operator schemas (`TypeAndShapeInferenceFunction`) together with auxiliary helpers such as `propagateShapeFromInputToOutput` and `propagateElemTypeFromInputToOutput`. These implementations:

- are written in C++
- are not machine-readable specifications
- cannot easily be analyzed or reused
- vary across implementations
- are difficult to test formally

A declarative specification of type and shape rules enables:

- consistent implementations across runtimes
- automatic validation of operator definitions
- symbolic shape reasoning
- explicit element type propagation
- improved tooling (compilers, optimizers, converters)
- formal reasoning about graph transformations

OTSL provides a compact and expressive language for describing these rules.

---

## 2 Goals

The proposed DSL must support:

1. symbolic dimension propagation
2. tensor element type propagation
3. constraints between dimensions and types
4. broadcast semantics
5. partial inference
6. shape tensor evaluation
7. rank reasoning
8. operator attributes influencing types and shapes

The DSL must remain:

- deterministic
- non-Turing-complete
- statically verifiable

---

## 3 Non-Goals

OTSL does not attempt to:

- describe full tensor computation semantics
- evaluate arbitrary tensor values
- replace ONNX operator definitions
- introduce general programming constructs

---

## 4 Terminology

| Term        | Meaning                                      |
|-------------|----------------------------------------------|
| Rank        | number of tensor dimensions                  |
| Dim         | single tensor dimension                      |
| Shape       | ordered list of dimensions                   |
| Type        | tensor element type                          |
| ShapeTensor | tensor whose statically available integer contents encode shape-related values |

---

## 5 Type System

OTSL defines the following primitive types.

### Rank

Non-negative integer.

```
rank(A)
```

### Dim

A dimension term may be:

- constant integer
- symbolic variable
- expression over dimensions
- unknown
- unknown non-negative

Examples:

```
32
sym("N")
dim(A,0) + 5
?
unknown_nonnegative()
```

OTSL uses the following dimension state model.

| State                | Notation                  | Meaning |
|---------------------|---------------------------|---------|
| known constant      | `32`                      | exact non-negative integer dimension |
| symbolic variable   | `sym("N")`                | named unknown dimension participating in equality reasoning |
| expression          | `dim(A,0) + 5`            | symbolic term over dimension values |
| unknown             | `?`                       | fresh unconstrained dimension term with no reusable identity |
| unknown non-negative| `unknown_nonnegative()`   | fresh unknown term additionally constrained to be `>= 0` |

Dimensions appearing in tensor shapes denote non-negative runtime sizes. `unknown_nonnegative()` is used when the specification knows only that a data-dependent result is non-negative. Plain `?` indicates no useful arithmetic information beyond the fact that the value is a dimension term.

### Shape

Ordered list of dimensions.

Example:

```
[dim(A,0), dim(B,1)]
```

### Type

Tensor element type.

Examples:

```
float
double
int64
type(A)
```

Type expressions may be compared in constraints and used in result assignments.

Examples:

```
type(X)
type(X) == type(Y)
type(X) in {float, double}
```

### Bool

Constraint expressions.

Examples:

```
dim(A,0) == dim(B,0)
type(X) == type(Y)
type(X) in {float, double}
```

### ShapeTensor

Tensor containing shape information.

Example:

```
shape_value(shape)
```

`shape_value()` reads from the abstract shape-tensor domain defined in section 10. Its result is not arbitrary tensor data and may include operator-defined sentinel integers such as those used by `Reshape`.

---

## 6 Core Syntax

A type and shape rule specification is embedded inside an operator definition.

Example:

```
rules {
  inputs  A, B;
  outputs Y;

  require rank(A) >= 2;
  require rank(B) >= 2;

  require dim(A,-1) == dim(B,-2);
  require type(A) == type(B);

  let batch = broadcast(prefix(A,-2), prefix(B,-2));

  result Y.shape = concat(batch, [dim(A,-2), dim(B,-1)]);
  result Y.type  = type(A);
}
```

---

## 7 Statements

### `require`

Declares a constraint that must hold.

Examples:

```
require dim(A,1) == dim(B,0);
require type(X) == type(Y);
require type(X) in {float, double};
```

Violation indicates an invalid graph.

---

### `let`

Declares an intermediate symbolic value.

```
let m = dim(A,-2);
```

---

### `result`

Defines the shape and/or type of an output tensor.

```
result Y.shape = [dim(A,0), dim(B,1)];
result Y.type  = type(A);
```

---

### `when`

Defines conditional rules.

```
when rank(A) == 1 and rank(B) == 1 {
  result Y.shape = [];
  result Y.type  = type(A);
}
```

Multiple branches may exist. Conditions are evaluated in source order using the current environment produced by preceding active statements in the enclosing scope.

A `when` condition has three-valued semantics: `true`, `false`, or `unknown`. Only `true` activates the branch. `false` and `unknown` do not activate it.

Every active branch is evaluated; `when` is not first-match. Active branches contribute additional `let` bindings (branch-local), constraints, and result assignments. If multiple active branches assign the same output field, the normalized assigned expressions must be identical; otherwise the rule result is `invalid`.

---

### `shape_tensors`

Declares which inputs may be inspected by `shape_value()`.

```
shape_tensors shape;
```

`shape_tensors` names a subset of `inputs`. The declaration is part of the operator schema and is therefore statically known.

---

## 8 Dimension Expressions

Supported operations:

```
d1 + d2
d1 - d2
d1 * d2
floordiv(d1,d2)
ceildiv(d1,d2)
max(d1,d2)
min(d1,d2)
```

Conditional and bounded structural expressions used by the examples are also part of the DSL:

```
if c then e1 else e2
range(n)
map x in S: e
sum(map x in S: e)
```

These expressions operate on `Dim`, `Shape`, or finite structural sequences. `map`, `range`, and `sum` are declarative structural operators, not general iteration constructs.

Their domains must be finite and statically delimited by the schema or by already-computed ranks or sequence lengths. They introduce no mutation, recursion, or user-defined control flow. If an implementation cannot establish the iteration domain during inference, the enclosing expression evaluates conservatively to `unknown` rather than triggering arbitrary computation.

Dimension expressions are first-order terms. Implementations may normalize them only by substitution of solved equalities and deterministic simplifications such as constant folding and elimination of identity operations. General algebraic rearrangement is not required.

---

## 9 Shape Operators

The DSL provides built-in shape manipulation primitives.

### `shape`

```
shape(A)
```

Returns the shape of tensor A.

---

### `dim`

```
dim(A,i)
```

Returns dimension `i`. Negative indices count from the end.

---

### `prefix` / `suffix`

```
prefix(A,k)
suffix(A,k)
```

Return slices of shapes.

---

### `concat`

```
concat(shape1, shape2)
```

Concatenates shapes.

---

### `broadcast`

```
broadcast(shape1, shape2)
```

Computes broadcasted shape.

---

### `normalize_axis`

```
normalize_axis(axis, rank)
```

Normalizes a possibly negative axis into the interval `[0, rank-1]`.

---

### `permute`

```
permute(shape(A), perm)
```

Applies dimension permutation.

---

### `resolve_reshape`

```
resolve_reshape(input_shape, target_shape)
```

Computes the output shape produced by the ONNX `Reshape` operator from an input shape and a shape-tensor value.

`broadcast`, `normalize_axis`, and `resolve_reshape` are normative built-ins. Their semantics follow the corresponding ONNX operator definitions and helper behavior for the applicable opset. Implementations may reuse existing ONNX helper code, but they must not substitute implementation-defined behavior.

---

## 10 Shape Tensor Evaluation

Some operators accept tensors describing shapes.

Example:

```
shape_tensors shape;
let target = shape_value(shape);
```

`shape_value()` extracts values from the abstract shape-tensor domain, not from arbitrary runtime tensor computation.

A shape tensor input is declared with `shape_tensors` and must be an integer-typed input designated by the operator schema as carrying shape information.

`shape_value(x)` is valid only when `x` is a direct input reference declared in `shape_tensors`. It is not valid on arbitrary expressions, intermediate tensors, or outputs of general computation.

The value returned by `shape_value(x)` is a finite abstract integer sequence whose entries may be known integers or unknown entries. The sequence may contain operator-defined sentinel integers when permitted by ONNX semantics; for example, the `shape` input of `Reshape` may contain `0` or `-1`.

OTSL does not evaluate arbitrary tensor values. An implementation may populate the abstract shape-tensor value only from sources already available to ONNX inference, such as constant initializers, constant-folded shape carriers, or other schema-defined shape tensors. If the contents are unavailable, `shape_value(x)` yields a sequence of unknown entries of the appropriate abstract length.

---

## 11 Partial Inference

OTSL supports incomplete knowledge for both types and shapes.

Possible dimension states:

| State                | Example                   |
|---------------------|---------------------------|
| known constant      | `32`                      |
| symbolic variable   | `sym("N")`                |
| expression          | `sym("N") + 5`            |
| unknown             | `?`                       |
| unknown non-negative| `unknown_nonnegative()`   |

Constraints may propagate relationships even if values are unknown.

Example:

```
require dim(A,0) == dim(B,0)
```

Type inference may likewise remain unresolved when the rule block does not determine a unique element type.

---

## 12 Constraint Solving and Built-in Constraint Predicates

OTSL constraint solving is defined over normalized dimension terms. Let `normalize(t)` recursively substitute all solved equalities into `t` and apply deterministic local simplifications such as constant folding.

Equality constraints on dimensions are processed using the following rules:

1. `constant == constant` succeeds iff the constants are identical; otherwise the rule result is `invalid`.
2. `symbolic == symbolic`, `symbolic == unknown`, and `unknown == unknown` merge the participating representatives into one equivalence class. The canonical representative is the earliest syntactic occurrence in rule-block source order.
3. `representative == expression` binds the representative to the normalized expression iff the representative does not occur inside that expression. This substitution is then propagated to all later evaluations and result expressions.
4. `expression == expression` succeeds immediately when the normalized expressions are syntactically identical. If both normalize to different constants, the rule result is `invalid`. Otherwise the equality remains unresolved; implementations are not required to perform general symbolic algebra such as cancellation or rearrangement.

Each evaluation of `?` or `unknown_nonnegative()` produces a fresh dimension term. If such a term is unified with a constant, symbolic variable, or expression, that information propagates to every occurrence of that same term created through the current input binding, `let` binding, or equality propagation.

`unknown_nonnegative()` additionally contributes the constraint `d >= 0`. Any solved equality that forces such a term to a negative constant is a contradiction and therefore `invalid`.

Cyclic bindings are contradictions. For example, `require dim(A,0) == dim(A,0) + 1;` is `invalid`.

Constraint propagation is monotonic and deterministic. Implementations may refine terms by substitution, equivalence-class merging, and deterministic local simplification only. They must not rely on implementation-defined search, backtracking, or non-deterministic algebraic rewrites.

A `require` predicate evaluates in three-valued logic after normalization:

- `true`: the predicate is satisfied
- `false`: the active rule result is `invalid`
- `unknown`: the predicate remains unresolved and the overall inference result cannot be more precise than `partial`

```
compatible(d1,d2)
same_shape(A,B)
known(d)
```

`known(d)` is `true` iff `d` normalizes to a constant. `compatible(d1,d2)` is `false` only when the normalized terms are provably unequal, `true` when they are provably equal, and `unknown` otherwise. `same_shape(A,B)` is defined pointwise from rank equality and `compatible` on corresponding dimensions. Type comparisons use standard equality and set-membership expressions and are solved exactly; no subtype relation is introduced.

---

## 13 Approximation for Data-Dependent Operators

Some operators produce shapes depending on runtime data.

Example: `NonZero`.

```
rules {
  inputs X;
  outputs Y;

  result Y.shape = [rank(X), unknown_nonnegative()];
  result Y.type  = int64;
}
```

This provides a safe upper bound without evaluating tensor contents.

---

## 14 Example Specifications

### MatMul

```
rules {
  inputs A, B;
  outputs Y;

  require dim(A,-1) == dim(B,-2);
  require type(A) == type(B);

  let batch = broadcast(prefix(A,-2), prefix(B,-2));

  result Y.shape = concat(batch, [dim(A,-2), dim(B,-1)]);
  result Y.type  = type(A);
}
```

---

### Concat

```
rules {
  inputs Xs[];
  outputs Y;
  attributes axis;

  let ax = normalize_axis(axis, rank(Xs[0]));

  result Y.shape =
    map j in range(rank(Xs[0])):
      if j == ax
        then sum(map i in Xs: dim(i,j))
        else dim(Xs[0],j);

  result Y.type = type(Xs[0]);
}
```

The result type follows the variadic input family. Any additional homogeneous-type requirements may be enforced by the surrounding operator schema or explicit `require` constraints.

---

### Reshape

```
rules {
  inputs data, shape;
  shape_tensors shape;
  outputs reshaped;

  let target = shape_value(shape);

  result reshaped.shape = resolve_reshape(shape(data), target);
  result reshaped.type  = type(data);
}
```

`resolve_reshape` is a normative helper implementing ONNX reshape semantics.

---

## 15 Canonical Representation

Although the DSL may appear in human-readable form, the normative representation is a structured AST that includes both type and shape assignments.

Example (simplified JSON):

```json
{
  "statements":[
    {
      "kind":"require",
      "expr":{
        "op":"eq",
        "args":[
          {"op":"type","tensor":"X"},
          {"op":"type","tensor":"slope"}
        ]
      }
    },
    {
      "kind":"result_shape",
      "tensor":"Y",
      "expr":{"op":"shape_of","tensor":"X"}
    },
    {
      "kind":"result_type",
      "tensor":"Y",
      "expr":{"op":"type_of","tensor":"X"}
    }
  ]
}
```

The textual syntax is considered a presentation format. The AST must also encode declarations such as variadic inputs, attributes, and `shape_tensors`.

### C++ Shape and Type Rule Representation

OTSL text is only a presentation format. The canonical representation used by ONNX schemas is an AST embedded directly in the C++ `OpSchema` definition.

Example:

```cpp
OpSchema()
  .SetTypeAndShapeRules(
      Require(
          Broadcastable(
              ShapeOf("slope"),
              ShapeOf("X")
          )
      ),
      Require(
          TypeEq("X", "slope")
      ),
      ResultShape(
          "Y",
          ShapeOf("X")
      ),
      ResultType(
          "Y",
          TypeOf("X")
      )
  );
```

The OTSL textual syntax maps directly to this AST and remains the human-readable way to author and review rule blocks.

---

## 16 Formal Evaluation Model

Evaluation of a rule block is deterministic and proceeds as follows:

1. Bind the schema-declared inputs, outputs, attributes, variadic families, and `shape_tensors`.
2. Initialize the environment with the available input ranks, shapes, element types, and shape-tensor abstract values. Missing information is represented by fresh unknown terms from section 5.
3. Evaluate top-level statements in source order.
4. For each active `let`, evaluate the right-hand side in the current environment, normalize it with the current solved equalities, and bind the name in the current lexical scope.
5. For each active `require`, normalize the predicate, add its constraints, and run the section 12 solver to a fixed point over the current accumulated constraints.
6. For each `when`, evaluate the condition in the current environment. If it is `true`, evaluate the nested statements in a child scope using the same algorithm; if it is `false` or `unknown`, skip the branch.
7. Accumulate all active `result` assignments. Multiple assignments to the same output field are permitted only if they normalize to the same expression; otherwise the rule result is `invalid`.
8. After all active statements have been processed, normalize every output shape and output type with the final solved equalities.
9. Classify the result:
   - `invalid` if any contradiction was derived
   - `exact` if every output type and dimension is fully known
   - `symbolic` if outputs are determined but contain symbolic variables or expressions and no unknowns
   - `partial` if some outputs are derived but contain unknowns or unresolved predicates
   - `unknown` if no output property can be derived

This algorithm defines the observable semantics. Implementations may optimize it, but any equivalent implementation must produce the same normalized constraints, output assignments, and status for the same inputs.

---

## 17 Error Semantics

Evaluation of type and shape rules may produce:

| Status   | Meaning                        |
|----------|--------------------------------|
| exact    | type and shape fully known     |
| symbolic | symbolic expressions present   |
| partial  | incomplete dimensions or types |
| unknown  | inference not possible         |
| invalid  | constraints violated           |

---

## 18 Integration with ONNX

Each `OpSchema` may optionally include a type and shape rule specification.

The canonical representation is a C++ AST attached to a specific operator schema version. The OTSL textual syntax maps directly to this AST and may be used for authoring, review, interchange, or generated documentation.

OTSL rules are versioned with the ONNX operator schema/opset version to which they are attached. A change in normative rule semantics requires the corresponding schema versioning discipline already used by ONNX.

If both an OTSL rule set and an imperative `TypeAndShapeInferenceFunction` are present for the same schema version, they are required to be semantically equivalent. OTSL is the normative portable specification; the imperative function is an allowed implementation strategy and backward-compatibility mechanism.

A runtime that does not support OTSL may ignore the attached rules and instead use an existing imperative inference function or perform no inference. Lack of OTSL support must not change model validity; it only affects the amount of inference available.

Example:

```cpp
OpSchema()
  .SetName("MatMul")
  .SetTypeAndShapeRules(...)
```

Existing imperative inference functions may remain for backward compatibility.

---

## 19 Reference Implementation Goals

A prototype implementation should:

- parse OTSL definitions
- propagate symbolic shapes and element types through graphs
- validate constraints
- support partial inference
- be deterministic

Suggested implementation languages:

- Python
- C++
- Rust

---

## 20 Backward Compatibility

This proposal does not modify ONNX graph semantics.

Existing models remain valid.

Runtimes may:

- ignore OTSL rules
- use them for validation
- use them for type and shape inference

---

## 21 Future Work

Possible extensions include:

- interval dimension reasoning
- richer constraint solving beyond equality and local normalization
- automated operator verification
- integration with graph optimization passes

---

## 22 Conclusion

OTSL introduces a declarative, machine-readable way to describe ONNX operator type and shape semantics.

This approach improves portability, tooling support, and formal verification while remaining compatible with existing ONNX infrastructure.
