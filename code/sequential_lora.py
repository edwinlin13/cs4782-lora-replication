import math
import torch
import torch.nn as nn


class LoRAQVStage(nn.Module):
    """A single sequential stage's worth of LoRA params for the QV slices.

    Just holds (lora_q_A, lora_q_B, lora_v_A, lora_v_B) for one stage.
    The actual forward pass lives in LoRAQVStack which sums across stages.
    """

    def __init__(self, d_in, d_head, rank, alpha=None):
        super().__init__()
        if alpha is None:
            alpha = rank  # match the project convention: alpha=rank, scaling=1
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # same init as the regular LoRALinear: kaiming for A, zeros for B
        # so a fresh stage is a no-op (BA=0)
        self.lora_q_A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        self.lora_q_B = nn.Parameter(torch.zeros(d_head, rank))

        self.lora_v_A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
        self.lora_v_B = nn.Parameter(torch.zeros(d_head, rank))


class LoRAQVStack(nn.Module):
    """Sequential analog of LoRAQVWrapper.

    Wraps GPT-2's fused c_attn (Conv1D) and holds a ModuleList of LoRAQVStage
    modules. Forward pass sums all stage contributions on the Q and V slices.

    Mathematically equivalent to growing a single (A, B) by appending rows/cols,
    but keeping stages separate lets us control requires_grad and LR per stage.
    """

    def __init__(self, original_c_attn, rank, alpha):
        super().__init__()
        self.original_c_attn = original_c_attn

        # freeze the underlying c_attn weights (same as LoRAQVWrapper)
        for p in self.original_c_attn.parameters():
            p.requires_grad = False

        self.d_in = original_c_attn.weight.shape[0]   # 768
        self.d_head = original_c_attn.nf // 3          # 768

        # start with one stage
        self.stages = nn.ModuleList([
            LoRAQVStage(self.d_in, self.d_head, rank=rank, alpha=alpha)
        ])

    def append_stage(self, rank, alpha=None):
        """Add a new LoRA stage. B=0 init means it's a no-op at append time."""
        stage = LoRAQVStage(self.d_in, self.d_head, rank=rank, alpha=alpha)
        # move new stage onto the same device as the others (in case stack was moved to gpu)
        device = self.original_c_attn.weight.device
        stage = stage.to(device)
        self.stages.append(stage)
        return stage

    def forward(self, x):
        qkv = self.original_c_attn(x)
        d_head = qkv.shape[-1] // 3

        # accumulate Q and V deltas across all stages
        # we clone qkv first to avoid in-place issues w/ autograd
        qkv = qkv.clone()
        for stage in self.stages:
            lora_q = (x @ stage.lora_q_A.T @ stage.lora_q_B.T) * stage.scaling
            lora_v = (x @ stage.lora_v_A.T @ stage.lora_v_B.T) * stage.scaling
            qkv[:, :, :d_head] += lora_q
            qkv[:, :, 2 * d_head:] += lora_v
        return qkv

    @property
    def total_rank(self):
        return sum(stage.rank for stage in self.stages)
