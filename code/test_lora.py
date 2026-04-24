import torch
import torch.nn as nn
from lora import LoRALinear, LoRAQVWrapper, inject_lora


def test_lora_linear_output_shape():
    """LoRALinear should produce the same output shape as the original layer."""
    linear = nn.Linear(768, 768, bias=True)
    lora = LoRALinear(linear, rank=4, alpha=4)
    x = torch.randn(2, 10, 768)
    out = lora(x)
    assert out.shape == (2, 10, 768)


def test_lora_linear_initial_output_matches_original():
    """At initialization (B=0), LoRA output should equal the original layer output."""
    linear = nn.Linear(768, 768, bias=True)
    lora = LoRALinear(linear, rank=4, alpha=4)
    x = torch.randn(2, 10, 768)
    with torch.no_grad():
        original_out = linear(x)
        lora_out = lora(x)
    assert torch.allclose(original_out, lora_out, atol=1e-6)


def test_lora_linear_only_lora_params_trainable():
    """Only A and B matrices should have requires_grad=True."""
    linear = nn.Linear(768, 768, bias=True)
    lora = LoRALinear(linear, rank=4, alpha=4)
    trainable = {name for name, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}


def test_lora_linear_parameter_shapes():
    """A should be (rank, d_in), B should be (d_out, rank)."""
    linear = nn.Linear(768, 256, bias=True)
    lora = LoRALinear(linear, rank=8, alpha=8)
    assert lora.lora_A.shape == (8, 768)
    assert lora.lora_B.shape == (256, 8)


def test_lora_linear_b_initialized_to_zero():
    """B matrix should be initialized to zeros so LoRA starts as identity."""
    linear = nn.Linear(768, 768, bias=True)
    lora = LoRALinear(linear, rank=4, alpha=4)
    assert torch.all(lora.lora_B == 0)


def test_qv_wrapper_output_shape():
    """QV wrapper should produce same shape as original c_attn."""
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    c_attn = model.transformer.h[0].attn.c_attn
    wrapper = LoRAQVWrapper(c_attn, rank=4, alpha=4)
    x = torch.randn(2, 10, 768)
    out = wrapper(x)
    assert out.shape == (2, 10, 2304)


def test_qv_wrapper_initial_output_matches_original():
    """At initialization, QV wrapper output should match original c_attn."""
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    c_attn = model.transformer.h[0].attn.c_attn
    x = torch.randn(2, 10, 768)
    with torch.no_grad():
        original_out = c_attn(x)
    wrapper = LoRAQVWrapper(c_attn, rank=4, alpha=4)
    with torch.no_grad():
        wrapper_out = wrapper(x)
    assert torch.allclose(original_out, wrapper_out, atol=1e-6)


def test_inject_lora_freezes_base_params():
    """After injection, only LoRA parameters should be trainable."""
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_lora(model, rank=4, alpha=4)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    for name in trainable_names:
        assert "lora_" in name, f"Non-LoRA param is trainable: {name}"
    assert len(trainable_names) > 0, "No trainable parameters found"


def test_inject_lora_correct_num_wrappers():
    """GPT-2 small has 12 layers, each gets a QV wrapper = 12 wrappers."""
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_lora(model, rank=4, alpha=4)
    wrapper_count = sum(1 for _, m in model.named_modules() if isinstance(m, LoRAQVWrapper))
    assert wrapper_count == 12, f"Expected 12 LoRAQVWrapper modules, got {wrapper_count}"


if __name__ == "__main__":
    test_lora_linear_output_shape()
    test_lora_linear_initial_output_matches_original()
    test_lora_linear_only_lora_params_trainable()
    test_lora_linear_parameter_shapes()
    test_lora_linear_b_initialized_to_zero()
    print("All LoRALinear tests passed!")
    print("Running injection tests (requires downloading GPT-2)...")
    test_qv_wrapper_output_shape()
    test_qv_wrapper_initial_output_matches_original()
    test_inject_lora_freezes_base_params()
    test_inject_lora_correct_num_wrappers()
    print("All injection tests passed!")
