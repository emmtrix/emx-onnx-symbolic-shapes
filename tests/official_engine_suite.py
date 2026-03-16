"""Shared helpers for official ONNX vs OTSL shape-inference comparisons."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, TypeProto, numpy_helper, shape_inference
from onnx.backend.test.loader import load_model_tests

from oscl.engine import OsclShapeInferenceEngine

OFFICIAL_TEST_KINDS = (
    "node",
    "real",
    "simple",
    "pytorch-converted",
    "pytorch-operator",
)
EXPECTED_RESULTS_PATH = (
    Path(__file__).resolve().parent / "data" / "test_engine_expected.json"
)


@dataclass(frozen=True)
class OfficialTestCase:
    """Minimal metadata needed to load and compare one official ONNX test."""

    case_id: str
    kind: str
    name: str
    model_name: str
    model_dir: str | None
    url: str | None


ENGINE = OsclShapeInferenceEngine()


def collect_official_test_cases() -> list[OfficialTestCase]:
    """Return the full official ONNX backend test suite from on-disk data."""
    cases: list[OfficialTestCase] = []
    for kind in OFFICIAL_TEST_KINDS:
        loaded = sorted(load_model_tests(kind=kind), key=lambda case: case.name)
        for case in loaded:
            cases.append(
                OfficialTestCase(
                    case_id=f"{kind}::{case.name}",
                    kind=kind,
                    name=case.name,
                    model_name=(
                        Path(case.model_dir).name
                        if case.model_dir is not None
                        else case.model_name
                    ),
                    model_dir=case.model_dir,
                    url=case.url,
                )
            )
    return cases


def load_expected_results(path: Path = EXPECTED_RESULTS_PATH) -> dict[str, Any]:
    """Load the JSON document containing expected comparison results."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_expected_results_document() -> dict[str, Any]:
    """Build an extensible JSON document for all official ONNX tests."""
    cases = collect_official_test_cases()
    tests: dict[str, Any] = {}
    for case in cases:
        tests[case.case_id] = {
            "kind": case.kind,
            "name": case.name,
            "model_name": case.model_name,
            "expected": {
                "comparison_result": compare_case(case),
            },
        }

    return {
        "schema_version": 1,
        "suite": {
            "source": "onnx.backend.test",
            "onnx_version": onnx.__version__,
            "case_count": len(cases),
            "kinds": list(OFFICIAL_TEST_KINDS),
        },
        "tests": tests,
    }


def compare_case(case: OfficialTestCase) -> str:
    """Return ``OK`` or a deterministic error string for one official test."""
    try:
        model = _load_model(case)
        input_values = _load_first_input_data_set(model, case)
        enriched_model = _inject_constant_inputs(model, input_values)

        onnx_inferred = shape_inference.infer_shapes(copy.deepcopy(enriched_model))
        oscl_inferred = ENGINE.infer_shapes(enriched_model)

        onnx_outputs = _get_output_signatures(onnx_inferred)
        oscl_outputs = _get_output_signatures(oscl_inferred)
        mismatch = _first_output_mismatch(onnx_outputs, oscl_outputs)
        if mismatch is not None:
            return mismatch
        return "OK"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def get_output_shapes(model: ModelProto) -> dict[str, list[int | str]]:
    """Extract tensor output shapes as ``{name: [dims]}``."""
    result: dict[str, list[int | str]] = {}
    for out in model.graph.output:
        kind = out.type.WhichOneof("value")
        if kind != "tensor_type":
            continue
        tp = out.type.tensor_type
        if tp.HasField("shape"):
            result[out.name] = [_dim_signature(dim) for dim in tp.shape.dim]
    return result


def get_output_types(model: ModelProto) -> dict[str, int]:
    """Extract tensor output element types as ``{name: elem_type}``."""
    result: dict[str, int] = {}
    for out in model.graph.output:
        if out.type.HasField("tensor_type"):
            elem_type = out.type.tensor_type.elem_type
            if elem_type != TensorProto.UNDEFINED:
                result[out.name] = elem_type
    return result


def _load_model(case: OfficialTestCase) -> ModelProto:
    model_path = _resolve_model_path(case)
    return onnx.load(model_path)


def _resolve_model_path(case: OfficialTestCase) -> Path:
    if case.model_dir is not None:
        return Path(case.model_dir) / "model.onnx"
    if case.url and case.url.startswith("onnx/backend/test/data/light/"):
        site_packages = Path(onnx.__file__).resolve().parent.parent
        return site_packages / case.url
    raise FileNotFoundError(
        f"Unable to resolve model path for official test {case.case_id!r}"
    )


def _load_first_input_data_set(
    model: ModelProto,
    case: OfficialTestCase,
) -> list[Any]:
    if case.model_dir is None:
        return []

    data_sets = sorted(Path(case.model_dir).glob("test_data_set_*"))
    if not data_sets:
        return []

    input_files = sorted(
        data_sets[0].glob("input_*.pb"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    values: list[Any] = []
    for index, input_file in enumerate(input_files):
        if index >= len(model.graph.input):
            break
        values.append(_load_proto_value(input_file, model.graph.input[index].type))
    return values


def _load_proto_value(proto_file: Path, type_proto: TypeProto) -> Any:
    payload = proto_file.read_bytes()
    if type_proto.HasField("tensor_type"):
        tensor = onnx.TensorProto()
        tensor.ParseFromString(payload)
        return numpy_helper.to_array(tensor)
    if type_proto.HasField("sequence_type"):
        sequence = onnx.SequenceProto()
        sequence.ParseFromString(payload)
        return numpy_helper.to_list(sequence)
    if type_proto.HasField("optional_type"):
        optional = onnx.OptionalProto()
        optional.ParseFromString(payload)
        return numpy_helper.to_optional(optional)
    if type_proto.HasField("map_type"):
        map_proto = onnx.MapProto()
        map_proto.ParseFromString(payload)
        return numpy_helper.to_dict(map_proto)
    raise TypeError(f"Unsupported input proto kind in {proto_file}")


def _inject_constant_inputs(
    model: ModelProto,
    input_arrays: list[Any],
) -> ModelProto:
    """Return a copy of *model* with shape-relevant tensor inputs as initializers."""
    max_float_init_elems = 16

    model = copy.deepcopy(model)
    graph = model.graph

    graph_inputs = list(graph.input)
    existing_init_names = {init.name for init in graph.initializer}

    for idx, inp in enumerate(graph_inputs):
        if idx >= len(input_arrays):
            break
        arr = input_arrays[idx]
        name = inp.name

        if isinstance(arr, np.generic) and not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if not isinstance(arr, np.ndarray):
            continue

        if name in existing_init_names:
            continue
        if arr.dtype.kind in ("i", "u"):
            graph.initializer.append(numpy_helper.from_array(arr, name=name))
        elif arr.dtype.kind == "f" and arr.size <= max_float_init_elems:
            graph.initializer.append(numpy_helper.from_array(arr, name=name))

    return model


def _first_output_mismatch(
    onnx_outputs: dict[str, dict[str, Any]],
    oscl_outputs: dict[str, dict[str, Any]],
) -> str | None:
    if set(onnx_outputs) != set(oscl_outputs):
        return (
            "output names differ: "
            f"OTSL={sorted(oscl_outputs)} ONNX={sorted(onnx_outputs)}"
        )

    for out_name in sorted(onnx_outputs):
        onnx_sig = onnx_outputs[out_name]
        oscl_sig = oscl_outputs[out_name]
        if oscl_sig != onnx_sig:
            return (
                f"output {out_name!r} mismatch: "
                f"OTSL={json.dumps(oscl_sig, sort_keys=True)} "
                f"ONNX={json.dumps(onnx_sig, sort_keys=True)}"
            )
    return None


def _get_output_signatures(model: ModelProto) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for out in model.graph.output:
        result[out.name] = _type_signature(out.type)
    return result


def _type_signature(type_proto: TypeProto) -> dict[str, Any]:
    if type_proto.HasField("tensor_type"):
        tensor_type = type_proto.tensor_type
        sig: dict[str, Any] = {
            "kind": "tensor",
            "elem_type": TensorProto.DataType.Name(tensor_type.elem_type),
        }
        if tensor_type.HasField("shape"):
            sig["shape"] = [_dim_signature(dim) for dim in tensor_type.shape.dim]
        return sig

    if type_proto.HasField("sequence_type"):
        return {
            "kind": "sequence",
            "elem": _type_signature(type_proto.sequence_type.elem_type),
        }

    if type_proto.HasField("optional_type"):
        return {
            "kind": "optional",
            "elem": _type_signature(type_proto.optional_type.elem_type),
        }

    if type_proto.HasField("map_type"):
        return {
            "kind": "map",
            "key_type": TensorProto.DataType.Name(type_proto.map_type.key_type),
            "value": _type_signature(type_proto.map_type.value_type),
        }

    if type_proto.HasField("sparse_tensor_type"):
        sparse_type = type_proto.sparse_tensor_type
        sig = {
            "kind": "sparse_tensor",
            "elem_type": TensorProto.DataType.Name(sparse_type.elem_type),
        }
        if sparse_type.HasField("shape"):
            sig["shape"] = [_dim_signature(dim) for dim in sparse_type.shape.dim]
        return sig

    return {"kind": "unknown"}


def _dim_signature(dim: onnx.TensorShapeProto.Dimension) -> int | str:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return "?"
