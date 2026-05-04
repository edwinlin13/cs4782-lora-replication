import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from sequential_lora import LoRAQVStage, LoRAQVStack
# inject_sequential_lora, append_stage_to_model, merge_sequential_lora imports
# come in later tasks once those functions exist


# we always use the same per-stage rank/alpha so scaling = 1
PER_STAGE_RANK = 2
PER_STAGE_ALPHA = 2


def _get_c_attn():
    """small helper, dont want to keep loading gpt2 over and over."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model, model.transformer.h[0].attn.c_attn


def test_stage_param_shapes():
    """LoRAQVStage A's are (rank, d_in), B's are (d_head, rank)."""
    _, c_attn = _get_c_attn()
    d_in = c_attn.weight.shape[0]
    d_head = c_attn.nf // 3
    stage = LoRAQVStage(d_in=d_in, d_head=d_head, rank=PER_STAGE_RANK)
    assert stage.lora_q_A.shape == (PER_STAGE_RANK, d_in)
    assert stage.lora_q_B.shape == (d_head, PER_STAGE_RANK)
    assert stage.lora_v_A.shape == (PER_STAGE_RANK, d_in)
    assert stage.lora_v_B.shape == (d_head, PER_STAGE_RANK)


def test_stage_b_initialized_to_zero():
    """B matrices should start at zero so a fresh stage is a no-op."""
    _, c_attn = _get_c_attn()
    d_in = c_attn.weight.shape[0]
    d_head = c_attn.nf // 3
    stage = LoRAQVStage(d_in=d_in, d_head=d_head, rank=PER_STAGE_RANK)
    assert torch.all(stage.lora_q_B == 0)
    assert torch.all(stage.lora_v_B == 0)


def test_stack_initial_output_matches_original():
    """With a single stage and B=0, stack output equals raw c_attn output."""
    _, c_attn = _get_c_attn()
    stack = LoRAQVStack(c_attn, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    x = torch.randn(2, 10, 768)
    with torch.no_grad():
        original = c_attn(x)
        out = stack(x)
    assert torch.allclose(original, out, atol=1e-6)


def test_stack_appended_stage_does_not_change_output():
    """Appending a fresh stage (B=0) must not change forward output."""
    _, c_attn = _get_c_attn()
    stack = LoRAQVStack(c_attn, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    # train the first stage a bit by injecting non-zero B
    with torch.no_grad():
        stack.stages[0].lora_q_B.normal_(0, 0.01)
        stack.stages[0].lora_v_B.normal_(0, 0.01)

    x = torch.randn(2, 10, 768)
    with torch.no_grad():
        before = stack(x)

    stack.append_stage(rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)

    with torch.no_grad():
        after = stack(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_stack_two_stages_sums_contributions():
    """With two trained stages, stack output equals sum of individual contributions."""
    _, c_attn = _get_c_attn()
    stack = LoRAQVStack(c_attn, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    stack.append_stage(rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)

    with torch.no_grad():
        for stage in stack.stages:
            stage.lora_q_B.normal_(0, 0.01)
            stage.lora_v_B.normal_(0, 0.01)

    x = torch.randn(2, 10, 768)
    d_head = c_attn.nf // 3

    with torch.no_grad():
        actual = stack(x)
        # compute expected manually
        expected = c_attn(x).clone()
        for stage in stack.stages:
            lora_q = (x @ stage.lora_q_A.T @ stage.lora_q_B.T) * stage.scaling
            lora_v = (x @ stage.lora_v_A.T @ stage.lora_v_B.T) * stage.scaling
            expected[:, :, :d_head] += lora_q
            expected[:, :, 2 * d_head:] += lora_v

    assert torch.allclose(actual, expected, atol=1e-5)


def test_stack_freezes_original_c_attn():
    """The wrapped c_attn must be frozen."""
    _, c_attn = _get_c_attn()
    stack = LoRAQVStack(c_attn, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    for p in stack.original_c_attn.parameters():
        assert not p.requires_grad


if __name__ == "__main__":
    test_stage_param_shapes()
    test_stage_b_initialized_to_zero()
    test_stack_initial_output_matches_original()
    test_stack_appended_stage_does_not_change_output()
    test_stack_two_stages_sums_contributions()
    test_stack_freezes_original_c_attn()
    print("Stack tests passed!")
