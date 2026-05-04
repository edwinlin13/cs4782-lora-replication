import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from sequential_lora import (
    LoRAQVStage,
    LoRAQVStack,
    inject_sequential_lora,
    append_stage_to_model,
)
# merge_sequential_lora import comes in task 4 once that fn exists


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


def test_inject_sequential_lora_freezes_base_params():
    """After injection, only LoRA stage params should be trainable."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_sequential_lora(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "stages." in name, f"non-stage param trainable: {name}"


def test_inject_sequential_lora_creates_one_stack_per_block():
    """GPT-2 small has 12 transformer blocks -> 12 LoRAQVStack modules."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_sequential_lora(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    stack_count = sum(1 for _, m in model.named_modules() if isinstance(m, LoRAQVStack))
    assert stack_count == 12, f"expected 12 stacks, got {stack_count}"


def test_inject_creates_one_stage_per_stack():
    """At injection time, each stack should have exactly one stage."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_sequential_lora(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    for _, m in model.named_modules():
        if isinstance(m, LoRAQVStack):
            assert len(m.stages) == 1


def test_append_stage_to_model_adds_to_every_stack():
    """append_stage_to_model adds a new stage to every LoRAQVStack."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_sequential_lora(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    new_stages = append_stage_to_model(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    assert len(new_stages) == 12  # one per stack
    for _, m in model.named_modules():
        if isinstance(m, LoRAQVStack):
            assert len(m.stages) == 2


def test_append_stage_returns_only_new_params():
    """The returned stages should be the freshly-added ones (B=0 still)."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_sequential_lora(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    new_stages = append_stage_to_model(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    for stage in new_stages:
        assert torch.all(stage.lora_q_B == 0)
        assert torch.all(stage.lora_v_B == 0)


def test_append_does_not_change_forward_output():
    """Appending a stage to a partially-trained model must not change logits."""
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_sequential_lora(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    # simulate training the first stage
    with torch.no_grad():
        for _, m in model.named_modules():
            if isinstance(m, LoRAQVStack):
                m.stages[0].lora_q_B.normal_(0, 0.01)
                m.stages[0].lora_v_B.normal_(0, 0.01)
    x = torch.randint(0, 50257, (1, 20))
    with torch.no_grad():
        before = model(x).logits
    append_stage_to_model(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    with torch.no_grad():
        after = model(x).logits
    assert torch.allclose(before, after, atol=1e-5)


if __name__ == "__main__":
    test_stage_param_shapes()
    test_stage_b_initialized_to_zero()
    test_stack_initial_output_matches_original()
    test_stack_appended_stage_does_not_change_output()
    test_stack_two_stages_sums_contributions()
    test_stack_freezes_original_c_attn()
    print("Stack tests passed!")
    print("Running injection tests (downloads gpt2)...")
    test_inject_sequential_lora_freezes_base_params()
    test_inject_sequential_lora_creates_one_stack_per_block()
    test_inject_creates_one_stage_per_stack()
    test_append_stage_to_model_adds_to_every_stack()
    test_append_stage_returns_only_new_params()
    test_append_does_not_change_forward_output()
    print("Injection and append tests passed!")
