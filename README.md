# emx-onnx-symbolic-shapes

Reference implementation for **OTSL - ONNX Type and Shape Language**, a declarative DSL for specifying shape inference rules for ONNX operators.

## Overview

This repository serves as the canonical reference implementation accompanying the OTSL RFC.
The goal is to provide a portable, deterministic engine that can:

- parse OTSL shape rule definitions
- propagate symbolic shapes through ONNX computation graphs
- validate shape constraints
- support partial / symbolic shape inference

## RFC

The full specification is available in [`docs/rfc-otsl.md`](docs/rfc-otsl.md).

## Status

Early draft / prototype stage. See the RFC for design details and open questions.

The current generated test status page is available at [`docs/test-engine-status.md`](docs/test-engine-status.md).

## Contributing

Contributions, feedback, and prototype implementations in Python, C++, or Rust are welcome.
Please open an issue or pull request and reference the relevant RFC section.
