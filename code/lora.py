import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for a frozen linear layer.

    Adds trainable matrices A (rank x d_in) and B (d_out x rank) such that:
        output = frozen_linear(x) + (x @ A^T @ B^T) * (alpha / rank)

    B is initialized to zeros so the LoRA contribution starts at zero.
    """

    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.original_layer = original_layer

        # freeze original weights so gradients dont flow back
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # took forever to figure out gpt2 uses Conv1D not Linear lol
        # Conv1D stores weights as (d_in, d_out) instead of (d_out, d_in)
        if hasattr(original_layer, 'nf'):
            # huggingface Conv1D: weight is (d_in, d_out), nf = d_out
            self.d_in = original_layer.weight.shape[0]
            self.d_out = original_layer.nf
            self.is_conv1d = True
        else:
            # regular nn.Linear: weight is (d_out, d_in)
            self.d_in = original_layer.in_features
            self.d_out = original_layer.out_features
            self.is_conv1d = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # A gets kaiming init, B gets zeros
        # this way the lora contribution starts at 0 and doesnt mess up the pretrained weights
        self.lora_A = nn.Parameter(torch.empty(rank, self.d_in))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(self.d_out, rank))

    def forward(self, x):
        # original frozen output + low rank update
        original_output = self.original_layer(x)
        # x @ A^T gives (batch, seq, rank), then @ B^T gives (batch, seq, d_out)
        lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original_output + lora_output
