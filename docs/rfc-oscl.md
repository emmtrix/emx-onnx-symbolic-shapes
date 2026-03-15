# RFC: Declarative Shape Rules for ONNX (OSCL)

## Status

Draft

## Authors

TBD

## Abstract

This document proposes a declarative domain-specific language (DSL) for specifying shape inference rules for ONNX operators. The language, called **ONNX Shape Constraint Language (OSCL)**, allows operator schemas to define input/output shape relationships using symbolic expressions and constraints.

The goal is to replace or complement existing imperative shape inference implementations with a machine-readable, declarative specification that:

- is deterministic
- supports symbolic dimensions
- supports partial shape inference
- is suitable for static analysis
- can be validated automatically

OSCL is intended to become a normative representation of operator shape semantics while remaining compatible with existing ONNX models and runtimes.

---

## 1 Motivation

Shape inference in ONNX is currently implemented as imperative functions attached to operator schemas (`TypeAndShapeInferenceFunction`). These implementations:

- are written in C++
- are not machine-readable specifications
- cannot easily be analyzed or reused
- vary across implementations
- are difficult to test formally

A declarative specification of shape rules enables:

- consistent implementations across runtimes
- automatic validation of operator definitions
- symbolic shape reasoning
- improved tooling (compilers, optimizers, converters)
- formal reasoning about graph transformations

OSCL provides a compact and expressive language for describing these rules.

---

## 2 Goals

The proposed DSL must support:

1. symbolic dimension propagation
2. constraints between dimensions
3. broadcast semantics
4. partial inference
5. shape tensor evaluation
6. rank reasoning
7. operator attributes influencing shapes

The DSL must remain:

- deterministic
- non-Turing-complete
- statically verifiable

---

## 3 Non-Goals

OSCL does not attempt to:

- describe full tensor computation semantics
- evaluate arbitrary tensor values
- replace ONNX operator definitions
- introduce general programming constructs

---

## 4 Terminology

| Term        | Meaning                                              |
|-------------|------------------------------------------------------|
| Rank        | number of tensor dimensions                          |
| Dim         | single tensor dimension                              |
| Shape       | ordered list of dimensions                           |
| ShapeTensor | tensor whose runtime values represent shapes         |

---

## 5 Type System

OSCL defines the following primitive types.

### Rank

Non-negative integer.

```
rank(A)
```

### Dim

A dimension may be:

- constant integer
- symbolic variable
- expression over dimensions
- unknown

Examples:

```
32
sym("N")
dim(A,0) + 5
?
```

### Shape

Ordered list of dimensions.

Example:

```
[dim(A,0), dim(B,1)]
```

### Bool

Constraint expressions.

Example:

```
dim(A,0) == dim(B,0)
```

### ShapeTensor

Tensor containing shape information.

Example:

```
shape_value(shape)
```

---

## 6 Core Syntax

A shape specification is embedded inside an operator definition.

Example:

```
shape {
  inputs  A, B;
  outputs Y;

  require rank(A) >= 2;
  require rank(B) >= 2;

  require dim(A,-1) == dim(B,-2);

  let batch = broadcast(prefix(A,-2), prefix(B,-2));

  result Y = concat(batch, [dim(A,-2), dim(B,-1)]);
}
```

---

## 7 Statements

### `require`

Declares a constraint that must hold.

```
require dim(A,1) == dim(B,0);
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

Defines the shape of an output tensor.

```
result Y = [dim(A,0), dim(B,1)];
```

---

### `when`

Defines conditional rules.

```
when rank(A) == 1 and rank(B) == 1 {
  result Y = [];
}
```

Multiple branches may exist.

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

These expressions operate on `Dim` values.

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

### `permute`

```
permute(shape(A), perm)
```

Applies dimension permutation.

---

## 10 Shape Tensor Evaluation

Some operators accept tensors describing shapes.

Example:

```
let target = shape_value(shape);
```

`shape_value()` extracts symbolic shape information.

Shape tensor evaluation is restricted to inputs explicitly defined as shape tensors in the operator specification.

---

## 11 Partial Inference

OSCL supports incomplete knowledge.

Possible dimension states:

| State            | Example  |
|------------------|----------|
| known constant   | `32`     |
| symbolic         | `sym("N")` |
| expression       | `N + 5`  |
| unknown          | `?`      |

Constraints may propagate relationships even if values are unknown.

Example:

```
require dim(A,0) == dim(B,0)
```

---

## 12 Built-in Constraint Predicates

```
compatible(d1,d2)
same_shape(A,B)
known(d)
```

These predicates allow safe reasoning when dimensions are partially known.

---

## 13 Approximation for Data-Dependent Operators

Some operators produce shapes depending on runtime data.

Example: `NonZero`.

```
shape {
  inputs X;
  outputs Y;

  result Y = [rank(X), unknown_nonnegative()];
}
```

This provides a safe upper bound without evaluating tensor contents.

---

## 14 Example Specifications

### MatMul

```
shape {
  inputs A, B;
  outputs Y;

  require dim(A,-1) == dim(B,-2);

  let batch = broadcast(prefix(A,-2), prefix(B,-2));

  result Y = concat(batch, [dim(A,-2), dim(B,-1)]);
}
```

---

### Concat

```
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

---

### Reshape

```
shape {
  inputs data, shape;
  outputs reshaped;

  let target = shape_value(shape);

  result reshaped = resolve_reshape(shape(data), target);
}
```

`resolve_reshape` is a normative helper implementing ONNX reshape semantics.

---

## 15 Canonical Representation

Although the DSL may appear in human-readable form, the normative representation should be a structured AST format.

Example (simplified JSON):

```json
{
  "inputs": ["A","B"],
  "outputs": ["Y"],
  "statements": [
    {
      "kind": "require",
      "expr": {
        "op": "eq",
        "args": [
          {"op": "dim", "args": ["A", -1]},
          {"op": "dim", "args": ["B", -2]}
        ]
      }
    }
  ]
}
```

The textual syntax is considered a presentation format.

---

## 16 Error Semantics

Evaluation of shape rules may produce:

| Status   | Meaning                         |
|----------|---------------------------------|
| exact    | shape fully known               |
| symbolic | symbolic expressions present    |
| partial  | incomplete dimensions           |
| unknown  | inference not possible          |
| invalid  | constraints violated            |

---

## 17 Integration with ONNX

Each `OpSchema` may optionally include a shape specification.

Example:

```cpp
OpSchema()
  .SetName("MatMul")
  .SetShapeRules(OSCL_definition)
```

Existing imperative inference functions may remain for backward compatibility.

---

## 18 Reference Implementation Goals

A prototype implementation should:

- parse OSCL definitions
- propagate symbolic shapes through graphs
- validate constraints
- support partial inference
- be deterministic

Suggested implementation languages:

- Python
- C++
- Rust

---

## 19 Backward Compatibility

This proposal does not modify ONNX graph semantics.

Existing models remain valid.

Runtimes may:

- ignore OSCL rules
- use them for validation
- use them for shape inference

---

## 20 Future Work

Possible extensions include:

- interval dimension reasoning
- constraint solving
- automated operator verification
- integration with graph optimization passes

---

## 21 Conclusion

OSCL introduces a declarative, machine-readable way to describe ONNX operator shape semantics.

This approach improves portability, tooling support, and formal verification while remaining compatible with existing ONNX infrastructure.
