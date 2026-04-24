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


class LoRAQVWrapper(nn.Module):
    """Wraps GPT-2's fused c_attn (Conv1D) and applies LoRA to Q and V slices only.

    GPT-2's c_attn projects (768 -> 2304), where output is [Q|K|V] each of dim 768.
    We only add LoRA to Q and V, not K (following the paper).
    """

    # this was pain to debug bc c_attn is a single layer that outputs Q K V concatenated
    # so we cant just wrap it with LoRALinear directly, we need to split the output
    # and add lora to just the Q and V parts

    def __init__(self, original_c_attn, rank, alpha):
        super().__init__()
        self.original_c_attn = original_c_attn

        for param in self.original_c_attn.parameters():
            param.requires_grad = False

        d_in = original_c_attn.weight.shape[0]   # 768
        d_head = original_c_attn.nf // 3          # 768 (each of Q, K, V)

        self.scaling = alpha / rank

        # separate lora matrices for Q and V
        self.lora_q_A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        self.lora_q_B = nn.Parameter(torch.zeros(d_head, rank))

        self.lora_v_A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
        self.lora_v_B = nn.Parameter(torch.zeros(d_head, rank))

    def forward(self, x):
        # get the original [Q|K|V] output
        qkv = self.original_c_attn(x)
        d_head = qkv.shape[-1] // 3

        # compute lora updates for Q and V only
        lora_q = (x @ self.lora_q_A.T @ self.lora_q_B.T) * self.scaling
        lora_v = (x @ self.lora_v_A.T @ self.lora_v_B.T) * self.scaling

        # add lora to the Q and V slices
        # this didnt work without clone() because of in-place modification issues
        qkv = qkv.clone()
        qkv[:, :, :d_head] += lora_q           # Q slice
        qkv[:, :, 2 * d_head:] += lora_v       # V slice (K is in the middle, untouched)

        return qkv


def inject_lora(model, rank, alpha, target_modules=None):
    """Inject LoRA into GPT-2's attention layers.

    Freezes everything, then replaces c_attn with LoRAQVWrapper
    so only the low-rank matrices are trainable.
    """
    if target_modules is None:
        target_modules = ["attn.c_attn"]

    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # collect modules to replace - cant modify dict while iterating
    # (learned this the hard way lol)
    replacements = []
    for name, module in model.named_modules():
        if any(name.endswith(target) for target in target_modules):
            replacements.append((name, module))

    for name, module in replacements:
        # walk down the module tree to find the parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]

        # use QV wrapper for c_attn, regular LoRALinear for everything else
        if name.endswith("c_attn"):
            wrapper = LoRAQVWrapper(module, rank, alpha)
        else:
            wrapper = LoRALinear(module, rank, alpha)

        setattr(parent, attr_name, wrapper)
