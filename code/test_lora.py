import torch
import torch.nn as nn
from lora import LoRALinear


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


if __name__ == "__main__":
    test_lora_linear_output_shape()
    test_lora_linear_initial_output_matches_original()
    test_lora_linear_only_lora_params_trainable()
    test_lora_linear_parameter_shapes()
    test_lora_linear_b_initialized_to_zero()
    print("All LoRALinear tests passed!")
