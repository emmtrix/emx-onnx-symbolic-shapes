"""Tests for the OSCL spec loader."""

from oscl.ast import ShapeSpec
from oscl.loader import load_all_specs, load_spec
import pytest


class TestLoadSpec:
    def test_load_matmul(self) -> None:
        spec = load_spec("matmul")
        assert isinstance(spec, ShapeSpec)
        assert len(spec.inputs) == 2
        assert spec.outputs == ["Y"]

    def test_load_concat(self) -> None:
        spec = load_spec("concat")
        assert isinstance(spec, ShapeSpec)
        assert spec.inputs[0].variadic is True

    def test_load_relu(self) -> None:
        spec = load_spec("relu")
        assert isinstance(spec, ShapeSpec)

    def test_load_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_spec("nonexistent_operator")


class TestLoadAllSpecs:
    def test_load_all(self) -> None:
        specs = load_all_specs()
        assert isinstance(specs, dict)
        expected_names = {
            "add", "concat", "flatten", "gather", "gemm",
            "matmul", "nonzero", "relu", "reshape", "softmax",
            "squeeze", "transpose", "unsqueeze",
        }
        assert expected_names.issubset(specs.keys())
        for name, spec in specs.items():
            assert isinstance(spec, ShapeSpec), f"{name} did not parse to ShapeSpec"

    def test_all_specs_have_outputs(self) -> None:
        specs = load_all_specs()
        for name, spec in specs.items():
            assert len(spec.outputs) >= 1, f"{name} has no outputs"
