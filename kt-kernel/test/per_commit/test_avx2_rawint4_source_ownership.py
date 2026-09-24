"""Source contract for AVX2 RAWINT4 per-expert weight ownership."""

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "operators"
    / "avx2"
    / "rawint4-moe.hpp"
)


def test_per_expert_rawint4_does_not_retain_safetensors_pointers():
    source = SOURCE.read_text(encoding="utf-8")
    per_expert = source.split("if (use_per_expert) {", 1)[1].split(
        "} else {", 1
    )[0]

    assert "owned_gate_weights_" in source
    assert "owned_up_weights_" in source
    assert "owned_down_weights_" in source
    assert "new uint8_t[owned_weight_bytes]" in per_expert
    assert "std::memcpy(new_gate_weights.get() + dst_offset" in per_expert
    assert "gate_bb_[expert_idx]->b = owned_gate_weights_.get() + offset" in per_expert
    assert "gate_bb_[expert_idx]->b = (uint8_t*)config_.gate_projs" not in per_expert


def test_rawint4_ownership_respects_cpu_expert_mask():
    source = SOURCE.read_text(encoding="utf-8")
    per_expert = source.split("if (use_per_expert) {", 1)[1].split(
        "} else {", 1
    )[0]

    assert "if (config_.should_skip_expert(logical_expert_id)) continue;" in per_expert
