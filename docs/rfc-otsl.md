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
| ShapeTensor | tensor whose runtime values represent shapes |

---

## 5 Type System

OTSL defines the following primitive types.

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

OTSL supports incomplete knowledge for both types and shapes.

Possible dimension states:

| State          | Example    |
|----------------|------------|
| known constant | `32`       |
| symbolic       | `sym("N")` |
| expression     | `N + 5`    |
| unknown        | `?`        |

Constraints may propagate relationships even if values are unknown.

Example:

```
require dim(A,0) == dim(B,0)
```

Type inference may likewise remain unresolved when the rule block does not determine a unique element type.

---

## 12 Built-in Constraint Predicates

```
compatible(d1,d2)
same_shape(A,B)
known(d)
```

These predicates allow safe reasoning when dimensions are partially known. Type comparisons use standard equality and set-membership expressions.

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

The textual syntax is considered a presentation format.

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

## 16 Error Semantics

Evaluation of type and shape rules may produce:

| Status   | Meaning                        |
|----------|--------------------------------|
| exact    | type and shape fully known     |
| symbolic | symbolic expressions present   |
| partial  | incomplete dimensions or types |
| unknown  | inference not possible         |
| invalid  | constraints violated           |

---

## 17 Integration with ONNX

Each `OpSchema` may optionally include a type and shape rule specification.

The canonical representation is a C++ AST attached to the schema. The OTSL textual syntax maps directly to this AST and may be used for authoring, review, interchange, or generated documentation. Runtimes may ignore these rules if unsupported.

Example:

```cpp
OpSchema()
  .SetName("MatMul")
  .SetTypeAndShapeRules(...)
```

Existing imperative inference functions may remain for backward compatibility.

---

## 18 Reference Implementation Goals

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

## 19 Backward Compatibility

This proposal does not modify ONNX graph semantics.

Existing models remain valid.

Runtimes may:

- ignore OTSL rules
- use them for validation
- use them for type and shape inference

---

## 20 Future Work

Possible extensions include:

- interval dimension reasoning
- constraint solving
- automated operator verification
- integration with graph optimization passes

---

## 21 Conclusion

OTSL introduces a declarative, machine-readable way to describe ONNX operator type and shape semantics.

This approach improves portability, tooling support, and formal verification while remaining compatible with existing ONNX infrastructure.
