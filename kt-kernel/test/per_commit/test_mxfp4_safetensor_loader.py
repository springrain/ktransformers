"""Regression tests for native MXFP4 safetensors schema detection."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

safetensors_torch = pytest.importorskip("safetensors.torch")

LOADER_PATH = Path(__file__).resolve().parents[2] / "python" / "utils" / "loader.py"
SPEC = importlib.util.spec_from_file_location("kt_mxfp4_loader_under_test", LOADER_PATH)
assert SPEC is not None and SPEC.loader is not None
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


PROJECTIONS = ("w1", "w3", "w2")


def _write_index(tmp_path: Path, weight_map: dict[str, str]) -> Path:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return index_path


def _write_checkpoint(
    tmp_path: Path,
    *,
    prefix: str,
    weight_suffix: str,
    scale_suffix: str,
    compressed_tensors_scale_shape: bool,
    projections: tuple[str, str, str] = PROJECTIONS,
    scale_as_float8_e8m0: bool = False,
    omit_key: str | None = None,
) -> None:
    tensors = {}
    for expert_id in range(2):
        for projection_id, projection in enumerate(projections):
            base = f"{prefix}.{expert_id}.{projection}"
            weight_key = f"{base}.{weight_suffix}"
            scale_key = f"{base}.{scale_suffix}"
            marker = 10 * expert_id + projection_id + 1
            tensors[weight_key] = torch.full((2, 32), marker, dtype=torch.uint8)
            scale = torch.full((2, 2), 127 + expert_id, dtype=torch.uint8)
            if compressed_tensors_scale_shape:
                scale = scale[:, None, :, None]
            if scale_as_float8_e8m0:
                scale = scale.view(torch.float8_e8m0fnu)
            tensors[scale_key] = scale

    if omit_key is not None:
        tensors.pop(omit_key)
    safetensors_torch.save_file(tensors, tmp_path / "model.safetensors")


def _assert_loaded_experts(loaded) -> None:
    assert set(loaded) == {
        "gate",
        "up",
        "down",
        "gate_scale",
        "up_scale",
        "down_scale",
    }
    for projection_id, name in enumerate(("gate", "up", "down")):
        weights = loaded[name]
        scales = loaded[f"{name}_scale"]
        assert len(weights) == len(scales) == 2
        for expert_id, (weight, scale) in enumerate(zip(weights, scales)):
            marker = 10 * expert_id + projection_id + 1
            assert weight.dtype == torch.uint8
            assert weight.is_contiguous()
            assert tuple(weight.shape) == (2, 32)
            assert torch.all(weight == marker)
            assert scale.dtype == torch.bfloat16
            assert scale.is_contiguous()
            assert tuple(scale.shape) == (2, 2)
            assert torch.all(scale == float(2**expert_id))


def test_canonical_index_is_lazy_and_handles_reopen(tmp_path, monkeypatch):
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    safetensors_torch.save_file(
        {"tensor.a": torch.tensor([1.0]), "tensor.same_shard": torch.tensor([2.0])},
        shard_a,
    )
    safetensors_torch.save_file({"tensor.b": torch.tensor([3.0])}, shard_b)
    _write_index(
        tmp_path,
        {
            "tensor.a": shard_a.name,
            "tensor.same_shard": shard_a.name,
            "tensor.b": shard_b.name,
        },
    )
    # A canonical model index must isolate the model from unrelated files in
    # the same directory.
    safetensors_torch.save_file(
        {"adapter.tensor": torch.tensor([4.0])},
        tmp_path / "adapter_model.safetensors",
    )

    real_safe_open = loader.safe_open
    opened = []

    def tracking_safe_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(loader, "safe_open", tracking_safe_open)
    tensor_loader = loader.SafeTensorLoader(str(tmp_path))

    assert opened == []
    assert tensor_loader.file_handle_map == {}
    assert "adapter.tensor" not in tensor_loader.tensor_file_map
    assert torch.equal(tensor_loader.load_tensor("tensor.a"), torch.tensor([1.0]))
    assert opened == [shard_a.name]
    assert torch.equal(
        tensor_loader.load_tensor("tensor.same_shard"), torch.tensor([2.0])
    )
    assert opened == [shard_a.name]

    tensor_map_id = id(tensor_loader.tensor_file_map)
    tensor_loader.close_all_handles(collect=False)
    assert tensor_loader.file_handle_map == {}
    assert id(tensor_loader.tensor_file_map) == tensor_map_id
    assert torch.equal(tensor_loader.load_tensor("tensor.b"), torch.tensor([3.0]))
    assert opened == [shard_a.name, shard_b.name]


def test_explicit_safetensors_file_does_not_scan_siblings(tmp_path):
    selected = tmp_path / "selected.safetensors"
    unrelated = tmp_path / "unrelated.safetensors"
    safetensors_torch.save_file({"selected": torch.tensor([1])}, selected)
    safetensors_torch.save_file({"unrelated": torch.tensor([2])}, unrelated)

    tensor_loader = loader.SafeTensorLoader(str(selected))

    assert set(tensor_loader.tensor_file_map) == {"selected"}


def test_fallback_uses_full_paths_for_duplicate_basenames(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    safetensors_torch.save_file(
        {"left.tensor": torch.tensor([1])}, left / "model.safetensors"
    )
    safetensors_torch.save_file(
        {"right.tensor": torch.tensor([2])}, right / "model.safetensors"
    )

    tensor_loader = loader.SafeTensorLoader(str(tmp_path))

    assert torch.equal(tensor_loader.load_tensor("left.tensor"), torch.tensor([1]))
    assert torch.equal(tensor_loader.load_tensor("right.tensor"), torch.tensor([2]))
    assert len(set(tensor_loader.tensor_file_map.values())) == 2


def test_index_missing_shard_fails_during_initialization(tmp_path):
    missing_name = "model-00001-of-00001.safetensors"
    _write_index(tmp_path, {"tensor": missing_name})

    with pytest.raises(FileNotFoundError, match="missing shard"):
        loader.SafeTensorLoader(str(tmp_path))


def test_format_detection_scans_beyond_first_thousand_index_keys(tmp_path):
    shard = tmp_path / "model.safetensors"
    safetensors_torch.save_file({"placeholder": torch.tensor([1.0])}, shard)
    weight_map = {f"non_expert.{idx}": shard.name for idx in range(1001)}
    weight_map[
        "model.layers.1.block_sparse_moe.experts.0.w1.weight"
    ] = shard.name
    _write_index(tmp_path, weight_map)

    tensor_loader = loader.FP8SafeTensorLoader(
        str(tmp_path), scale_suffix="weight_scale_inv"
    )

    assert tensor_loader._detected_format == "mixtral"


@pytest.mark.parametrize("loader_kind", ["fp8", "bf16"])
def test_format_specific_loaders_reopen_lazy_handles(tmp_path, loader_kind):
    prefix = "model.layers.1.mlp.experts.0"
    tensors = {
        f"{prefix}.gate_proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        f"{prefix}.up_proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        f"{prefix}.down_proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
    }
    if loader_kind == "fp8":
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"{prefix}.{projection}.weight_scale_inv"] = torch.ones((1, 1))
    checkpoint = tmp_path / "model.safetensors"
    safetensors_torch.save_file(tensors, checkpoint)

    if loader_kind == "fp8":
        tensor_loader = loader.FP8SafeTensorLoader(
            str(checkpoint), scale_suffix="weight_scale_inv"
        )
    else:
        tensor_loader = loader.BF16SafeTensorLoader(str(checkpoint))

    key = f"{prefix}.gate_proj.weight"
    tensor_loader.close_all_handles(collect=False)
    assert torch.equal(tensor_loader.load_tensor(key), tensors[key])


def test_ue8m0_to_bf16_preserves_minimum_value_and_nan_sentinel():
    raw = torch.tensor([0, 1, 126, 127, 128, 254, 255], dtype=torch.uint8)

    converted = loader.MXFP4SafeTensorLoader._ue8m0_to_bf16(raw)

    expected_bits = torch.tensor(
        [0x0040, 0x0080, 0x3F00, 0x3F80, 0x4000, 0x7F00, 0x7FC0],
        dtype=torch.int16,
    )
    assert torch.equal(converted.view(torch.int16), expected_bits)


@pytest.mark.parametrize(
    "prefix",
    [
        "language_model.layers.1.block_sparse_moe.experts",
        "language_model.model.layers.1.block_sparse_moe.experts",
    ],
)
def test_kimi_k3_raw_block_sparse_moe_layout(tmp_path, prefix):
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        weight_suffix="weight_packed",
        scale_suffix="weight_scale",
        compressed_tensors_scale_shape=True,
    )

    weights = loader.MXFP4SafeTensorLoader(str(tmp_path)).load_experts(
        "language_model.model.layers.1"
    )

    _assert_loaded_experts(weights)


def test_kimi_k3_mapped_mlp_layout(tmp_path):
    prefix = "language_model.model.layers.1.mlp.experts"
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        weight_suffix="weight_packed",
        scale_suffix="weight_scale",
        compressed_tensors_scale_shape=True,
    )

    weights = loader.MXFP4SafeTensorLoader(str(tmp_path)).load_experts(
        "language_model.model.layers.1"
    )

    _assert_loaded_experts(weights)


def test_compressed_projection_aliases_are_supported(tmp_path):
    prefix = "model.layers.1.mlp.experts"
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        projections=("gate_proj", "up_proj", "down_proj"),
        weight_suffix="weight_packed",
        scale_suffix="weight_scale",
        compressed_tensors_scale_shape=True,
    )

    weights = loader.MXFP4SafeTensorLoader(str(tmp_path)).load_experts("model.layers.1")

    _assert_loaded_experts(weights)


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.1.feed_forward.experts",
        "model.layers.1.experts",
    ],
)
def test_additional_common_expert_paths_are_supported(tmp_path, prefix):
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        projections=("gate_proj", "up_proj", "down_proj"),
        weight_suffix="weight_packed",
        scale_suffix="weight_scale",
        compressed_tensors_scale_shape=True,
    )

    weights = loader.MXFP4SafeTensorLoader(str(tmp_path)).load_experts("model.layers.1")

    _assert_loaded_experts(weights)


def test_group32_weight_and_weight_scale_are_supported(tmp_path):
    prefix = "model.layers.1.mlp.experts"
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        projections=("gate_proj", "up_proj", "down_proj"),
        weight_suffix="weight",
        scale_suffix="weight_scale",
        compressed_tensors_scale_shape=False,
    )

    weights = loader.MXFP4SafeTensorLoader(str(tmp_path)).load_experts("model.layers.1")

    _assert_loaded_experts(weights)


@pytest.mark.parametrize("weight_suffix", ["weight_packed", "weight"])
def test_weight_scale_inv_semantics_fail_fast(tmp_path, weight_suffix):
    prefix = "model.layers.1.mlp.experts"
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        projections=("gate_proj", "up_proj", "down_proj"),
        weight_suffix=weight_suffix,
        scale_suffix="weight_scale_inv",
        compressed_tensors_scale_shape=False,
    )

    mxfp4_loader = loader.MXFP4SafeTensorLoader(str(tmp_path))
    with pytest.raises(ValueError) as exc_info:
        mxfp4_loader.load_experts("model.layers.1")

    message = str(exc_info.value)
    assert "weight_scale_inv semantics are not accepted for MXFP4" in message
    assert f"gate_proj.{weight_suffix}" in message


def test_deepseek_v4_layout_remains_supported(tmp_path):
    prefix = "layers.1.ffn.experts"
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        weight_suffix="weight",
        scale_suffix="scale",
        compressed_tensors_scale_shape=False,
        scale_as_float8_e8m0=True,
    )

    weights = loader.MXFP4SafeTensorLoader(str(tmp_path)).load_experts("model.layers.1")

    _assert_loaded_experts(weights)


def test_missing_kimi_k3_tensor_reports_key_and_candidates(tmp_path):
    prefix = "language_model.layers.1.block_sparse_moe.experts"
    missing_key = f"{prefix}.1.w2.weight_scale"
    _write_checkpoint(
        tmp_path,
        prefix=prefix,
        weight_suffix="weight_packed",
        scale_suffix="weight_scale",
        compressed_tensors_scale_shape=True,
        omit_key=missing_key,
    )

    mxfp4_loader = loader.MXFP4SafeTensorLoader(str(tmp_path))
    with pytest.raises(ValueError) as exc_info:
        mxfp4_loader.load_experts("language_model.model.layers.1")

    message = str(exc_info.value)
    assert missing_key in message
    assert "Probed schema candidates" in message
    assert "block_sparse_moe.experts.0.w1.weight_packed" in message


def test_no_matching_schema_reports_all_kimi_k3_candidates(tmp_path):
    safetensors_torch.save_file(
        {"language_model.layers.1.input_layernorm.weight": torch.ones(2)},
        tmp_path / "model.safetensors",
    )
    mxfp4_loader = loader.MXFP4SafeTensorLoader(str(tmp_path))

    with pytest.raises(ValueError) as exc_info:
        mxfp4_loader.load_experts("language_model.model.layers.1")

    message = str(exc_info.value)
    assert (
        "language_model.layers.1.block_sparse_moe.experts.0.w1.weight_packed" in message
    )
    assert "language_model.layers.1.mlp.experts.0.w1.weight_packed" in message
    assert "language_model.model.layers.1.ffn.experts.0.w1.weight" in message


def test_incomplete_early_schema_does_not_shadow_a_complete_later_schema(tmp_path):
    tensors = {
        "model.layers.1.ffn.experts.0.w1.weight": torch.zeros(
            (2, 32), dtype=torch.uint8
        ),
        "model.layers.1.ffn.experts.0.w1.scale": torch.full(
            (2, 2), 127, dtype=torch.uint8
        ),
    }
    prefix = "model.layers.1.mlp.experts"
    for expert_id in range(2):
        for projection_id, projection in enumerate(PROJECTIONS):
            marker = 10 * expert_id + projection_id + 1
            base = f"{prefix}.{expert_id}.{projection}"
            tensors[f"{base}.weight_packed"] = torch.full(
                (2, 32), marker, dtype=torch.uint8
            )
            tensors[f"{base}.weight_scale"] = torch.full(
                (2, 1, 2, 1), 127 + expert_id, dtype=torch.uint8
            )
    safetensors_torch.save_file(tensors, tmp_path / "model.safetensors")

    loaded = loader.MXFP4SafeTensorLoader(str(tmp_path)).load_experts("model.layers.1")

    _assert_loaded_experts(loaded)


def test_nvfp4_group16_scale_is_not_misdetected_as_mxfp4(tmp_path):
    prefix = "model.layers.1.mlp.experts"
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{prefix}.0.{projection}"
        tensors[f"{base}.weight_packed"] = torch.zeros((2, 32), dtype=torch.uint8)
        tensors[f"{base}.weight_scale"] = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
    safetensors_torch.save_file(tensors, tmp_path / "model.safetensors")

    mxfp4_loader = loader.MXFP4SafeTensorLoader(str(tmp_path))
    with pytest.raises(ValueError) as exc_info:
        mxfp4_loader.load_experts("model.layers.1")

    message = str(exc_info.value)
    assert "Rejected matching candidates" in message
    assert "one E8M0 value per 32 FP4 values" in message


def test_e4m3_scale_dtype_is_not_accepted_as_e8m0(tmp_path):
    prefix = "model.layers.1.mlp.experts"
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{prefix}.0.{projection}"
        tensors[f"{base}.weight_packed"] = torch.zeros((2, 32), dtype=torch.uint8)
        tensors[f"{base}.weight_scale"] = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
    safetensors_torch.save_file(tensors, tmp_path / "model.safetensors")

    mxfp4_loader = loader.MXFP4SafeTensorLoader(str(tmp_path))
    with pytest.raises(ValueError) as exc_info:
        mxfp4_loader.load_experts("model.layers.1")

    message = str(exc_info.value)
    assert "Rejected matching candidates" in message
    assert "float8_e4m3fn" in message


def test_fp8_weight_dtype_is_not_misdetected_as_mxfp4(tmp_path):
    prefix = "model.layers.1.block_sparse_moe.experts"
    tensors = {}
    for projection in PROJECTIONS:
        base = f"{prefix}.0.{projection}"
        tensors[f"{base}.weight"] = torch.zeros((2, 64), dtype=torch.float8_e4m3fn)
        tensors[f"{base}.weight_scale"] = torch.full((2, 4), 127, dtype=torch.uint8)
    safetensors_torch.save_file(tensors, tmp_path / "model.safetensors")

    mxfp4_loader = loader.MXFP4SafeTensorLoader(str(tmp_path))
    with pytest.raises(ValueError) as exc_info:
        mxfp4_loader.load_experts("model.layers.1")

    message = str(exc_info.value)
    assert "Rejected matching candidates" in message
    assert "float8_e4m3fn" in message
