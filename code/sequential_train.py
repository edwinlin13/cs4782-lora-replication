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


class SequentialOptimizerManager:
    """AdamW + warmup-only LR + per-group LR multiplier for old stages.

    Two param groups (when alpha_old > 0):
        - "active": newly-appended stage's params, lr = base_lr * warmup_factor
        - "old":   all previously-appended stages' params, lr = alpha_old * base_lr * warmup_factor

    When alpha_old == 0, the old stages are frozen via requires_grad=False
    and removed from the optimizer entirely (saves moment state).

    Adam moments are preserved across stage transitions for old params -- we
    do NOT recreate the optimizer when adding a stage. We just shuffle the
    new stage's params into a freshly-added param group.
    """

    def __init__(self, model, base_lr, alpha_old, weight_decay, warmup_steps):
        self.model = model
        self.base_lr = base_lr
        self.alpha_old = alpha_old
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self._global_step = 0

        # initial state: one stage exists, everything trainable goes in the active group
        active_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            [{"params": active_params, "initial_lr": base_lr, "lr": base_lr,
              "_role": "active"}],
            lr=base_lr,
            weight_decay=weight_decay,
        )
        # apply warmup factor for step 0
        self.set_step(0)

    def _warmup_factor(self):
        if self.warmup_steps <= 0:
            return 1.0
        return min(self._global_step / self.warmup_steps, 1.0)

    def set_step(self, step):
        """Update each group's lr based on warmup. Call once per training step
        BEFORE optimizer.step()."""
        self._global_step = step
        f = self._warmup_factor()
        for g in self.optimizer.param_groups:
            g["lr"] = g["initial_lr"] * f

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def on_stage_added(self, new_stages):
        """Called immediately after append_stage_to_model.

        new_stages is a list of LoRAQVStage instances (the freshly-added ones).

        Reorganizes param groups:
          - alpha_old == 0: freeze old params (requires_grad=False), remove the
              "active" group, and create a new active group with only new params.
              Also remove old params from optimizer.state to free memory.
          - alpha_old > 0: rename the existing "active" group to "old" (with its
              initial_lr scaled by alpha_old), and add a new "active" group with
              the new stages' params at base_lr.
        """
        new_params = []
        for stage in new_stages:
            new_params.extend(stage.parameters())

        # take a template group dict so we get all of AdamW's required keys
        # (betas, eps, amsgrad, foreach, capturable, differentiable, fused, etc).
        # safer than hardcoding the list — survives torch version bumps.
        template = self.optimizer.param_groups[0]

        if self.alpha_old == 0.0:
            # freeze every existing param and remove from optimizer state
            for g in self.optimizer.param_groups:
                for p in g["params"]:
                    p.requires_grad = False
                    if p in self.optimizer.state:
                        del self.optimizer.state[p]
            # rebuild groups: just the new active group
            new_group = dict(template)
            new_group["params"] = new_params
            new_group["initial_lr"] = self.base_lr
            new_group["lr"] = self.base_lr
            new_group["_role"] = "active"
            new_group["weight_decay"] = self.weight_decay
            self.optimizer.param_groups = [new_group]
        else:
            # demote current active group(s) to "old" and add a new active group
            for g in self.optimizer.param_groups:
                if g.get("_role") == "active":
                    g["_role"] = "old"
                    g["initial_lr"] = self.base_lr * self.alpha_old
                # if there's already an "old" group, leave it as-is (its lr is
                # already alpha_old * base_lr from when IT was demoted)
            new_group = dict(template)
            new_group["params"] = new_params
            new_group["initial_lr"] = self.base_lr
            new_group["lr"] = self.base_lr
            new_group["_role"] = "active"
            new_group["weight_decay"] = self.weight_decay
            self.optimizer.add_param_group(new_group)

        # re-apply warmup factor since groups changed
        self.set_step(self._global_step)
