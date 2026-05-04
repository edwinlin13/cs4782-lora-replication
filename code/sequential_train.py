"""Sequential LoRA training: stage triggers, optimizer manager, training loop."""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW

from sequential_lora import (
    inject_sequential_lora,
    append_stage_to_model,
    merge_sequential_lora,
    LoRAQVStack,
    LoRAQVStage,
    get_all_stages,
)
from utils import count_parameters, save_metrics
from train import validate, generate_texts, compute_metrics


# ---- stage triggers ----

class FixedStepTrigger:
    """Fire stage transitions at evenly spaced step boundaries.

    For num_stages=k and total_steps=T, fires at T/k, 2T/k, ..., (k-1)T/k.
    The first stage is the one we start with (handled by inject_sequential_lora),
    so there are k-1 firings.
    """

    def __init__(self, total_steps, num_stages):
        self.total_steps = total_steps
        self.num_stages = num_stages
        if num_stages > 1:
            seg = total_steps // num_stages
            self.boundaries = set(seg * i for i in range(1, num_stages))
        else:
            self.boundaries = set()

    def update(self, step, **_):
        if step in self.boundaries:
            return {"stage": True}
        return {}


class PlateauTrigger:
    """Fire stage on val-loss plateau, stop on post-stage plateau or rank cap.

    State machine:
        WATCHING (default): if last `patience` evals show no improvement >delta
            and we haven't hit the rank cap, fire a stage and switch to GRACE.
            If we've hit the cap, signal stop.
        GRACE: just added a stage. Need to see at least one improvement within
            `patience` evals or we stop. Improvement -> back to WATCHING.

    "No improvement" = current val_loss has not decreased by more than `delta`
    relative to the best loss seen since the last stage transition (or run start).
    """

    def __init__(self, patience, delta, max_total_rank, per_stage_rank):
        self.patience = patience
        self.delta = delta
        self.max_total_rank = max_total_rank
        self.per_stage_rank = per_stage_rank
        # state
        self._best_since_marker = float("inf")
        self._evals_since_improvement = 0
        self._mode = "watching"  # or "grace"

    def update(self, val_loss, current_total_rank, **_):
        improved = val_loss < (self._best_since_marker - self.delta)
        if improved:
            self._best_since_marker = val_loss
            self._evals_since_improvement = 0
            # any improvement clears grace mode
            self._mode = "watching"
            return {}

        self._evals_since_improvement += 1
        if self._evals_since_improvement < self.patience:
            return {}

        # we hit patience without improvement
        if self._mode == "grace":
            # post-stage plateau -> give up
            return {"stop": True}

        # watching mode: try to add a stage if we have headroom
        next_rank = current_total_rank + self.per_stage_rank
        if next_rank > self.max_total_rank:
            return {"stop": True}

        # fire stage and enter grace mode
        self._best_since_marker = val_loss  # reset reference for the new stage
        self._evals_since_improvement = 0
        self._mode = "grace"
        return {"stage": True}
