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


def _stage_param_snapshot(model):
    """Capture the current state of every stage as a list-of-tensors snapshot."""
    snapshots = []
    for name, idx, stage in get_all_stages(model):
        snapshots.append({
            "stack": name,
            "stage_idx": idx,
            "lora_q_A": stage.lora_q_A.detach().cpu().clone(),
            "lora_q_B": stage.lora_q_B.detach().cpu().clone(),
            "lora_v_A": stage.lora_v_A.detach().cpu().clone(),
            "lora_v_B": stage.lora_v_B.detach().cpu().clone(),
        })
    return snapshots


def run_sequential_experiment(
    model,
    train_loader,
    val_loader,
    test_loader,
    test_dataset_hf,
    tokenizer,
    device,
    trigger,
    per_stage_rank=2,
    per_stage_alpha=2,
    alpha_old=1.0,
    num_epochs=5,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    eval_every=200,
    experiment_name="seq_experiment",
    checkpoint_dir="checkpoints/sequential",
    results_dir="../results/metrics/sequential",
    seed=42,
):
    """Run a sequential-LoRA experiment end-to-end.

    Assumes inject_sequential_lora has already been called on `model`.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    initial_params = count_parameters(model)

    optim_mgr = SequentialOptimizerManager(
        model=model,
        base_lr=learning_rate,
        alpha_old=alpha_old,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
    )

    train_loss_log = []   # list of (step, loss)
    val_loss_log = []     # list of (step, loss)
    stage_events = []     # list of dicts: {step, current_total_rank_after, reason, val_loss}
    stage_snapshots = []  # list of snapshots taken at each transition / final
    epoch_times = []

    global_step = 0
    total_steps_planned = num_epochs * len(train_loader)
    should_stop = False

    print(f"[{experiment_name}] starting, alpha_old={alpha_old}, "
          f"per_stage_rank={per_stage_rank}, total_planned_steps={total_steps_planned}")

    start_time = time.time()
    for epoch in range(num_epochs):
        if should_stop:
            break
        epoch_start = time.time()
        model.train()
        running_loss = 0
        running_count = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optim_mgr.set_step(global_step)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optim_mgr.zero_grad()
            loss.backward()
            optim_mgr.step()

            train_loss_log.append((global_step, loss.item()))
            running_loss += loss.item()
            running_count += 1
            global_step += 1

            # ---- step-based trigger update (fixed-step) ----
            decision = trigger.update(step=global_step, current_total_rank=_total_rank(model))
            if decision.get("stage"):
                _do_stage_transition(model, optim_mgr, stage_events, stage_snapshots,
                                     global_step, per_stage_rank, per_stage_alpha,
                                     reason="fixed_step", val_loss=None)
            if decision.get("stop"):
                should_stop = True
                break

            # ---- periodic validation + plateau trigger update ----
            if global_step % eval_every == 0:
                val_loss = _quick_validate(model, val_loader, device)
                val_loss_log.append((global_step, val_loss))
                model.train()  # _quick_validate puts it in eval mode

                decision = trigger.update(
                    step=global_step,
                    val_loss=val_loss,
                    current_total_rank=_total_rank(model),
                )
                if decision.get("stage"):
                    _do_stage_transition(model, optim_mgr, stage_events, stage_snapshots,
                                         global_step, per_stage_rank, per_stage_alpha,
                                         reason="plateau", val_loss=val_loss)
                if decision.get("stop"):
                    should_stop = True
                    break

        # end-of-epoch full validation (consistent w/ existing baselines for plotting)
        val_loss = _quick_validate(model, val_loader, device)
        val_loss_log.append((global_step, val_loss))
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_train = running_loss / max(running_count, 1)
        print(f"  Epoch {epoch+1}/{num_epochs} | step {global_step} "
              f"| avg train loss {avg_train:.4f} | val loss {val_loss:.4f} "
              f"| total_rank={_total_rank(model)} | time {epoch_elapsed:.1f}s",
              flush=True)

    total_train_time = time.time() - start_time

    # snapshot the final state of every stage before merging
    stage_snapshots.append({
        "step": global_step,
        "reason": "final",
        "snapshots": _stage_param_snapshot(model),
    })

    # ---- evaluation: merge then generate ----
    final_total_rank = _total_rank(model)
    merge_sequential_lora(model)
    print(f"  Generating on test set (post-merge, total rank = {final_total_rank})...")
    generated = generate_texts(model, test_dataset_hf, tokenizer, device)
    test_metrics = compute_metrics(generated, test_dataset_hf)
    print(f"  BLEU: {test_metrics['bleu']:.4f} | ROUGE-L: {test_metrics['rouge_l']:.4f}")

    results = {
        "experiment_name": experiment_name,
        "status": "complete",
        "config": {
            "alpha_old": alpha_old,
            "per_stage_rank": per_stage_rank,
            "per_stage_alpha": per_stage_alpha,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "eval_every": eval_every,
            "seed": seed,
        },
        "params": initial_params,
        "final_total_rank": final_total_rank,
        "train_loss_log": train_loss_log,
        "val_loss_log": val_loss_log,
        "stage_events": stage_events,
        "epoch_times": epoch_times,
        "total_train_time": total_train_time,
        "test_metrics": test_metrics,
    }
    save_metrics(results, f"{results_dir}/{experiment_name}.json")

    # save stage parameter snapshots (heavyweight, separate file)
    snap_path = f"{checkpoint_dir}/{experiment_name}_stage_snapshots.pt"
    os.makedirs(os.path.dirname(snap_path), exist_ok=True)
    torch.save(stage_snapshots, snap_path)

    return results


def _total_rank(model):
    """Sum of ranks across all stages of the first stack (all stacks are in lockstep)."""
    for _, m in model.named_modules():
        if isinstance(m, LoRAQVStack):
            return m.total_rank
    return 0


@torch.no_grad()
def _quick_validate(model, val_loader, device):
    """Lightweight validation pass -- just average cross-entropy loss."""
    model.eval()
    total = 0.0
    n = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total += out.loss.item()
        n += 1
    return total / max(n, 1)


def _do_stage_transition(model, optim_mgr, stage_events, stage_snapshots, step,
                         per_stage_rank, per_stage_alpha, reason, val_loss):
    """Take a snapshot of the about-to-be-demoted stage, then append + register."""
    # snapshot current state BEFORE adding the new stage
    stage_snapshots.append({
        "step": step,
        "reason": f"{reason}_pre_transition",
        "snapshots": _stage_param_snapshot(model),
    })
    new_stages = append_stage_to_model(model, rank=per_stage_rank, alpha=per_stage_alpha)
    optim_mgr.on_stage_added(new_stages)
    new_total = _total_rank(model)
    stage_events.append({
        "step": step,
        "reason": reason,
        "new_total_rank": new_total,
        "val_loss_at_transition": val_loss,
    })
    print(f"    >>> stage transition at step {step} ({reason}), "
          f"total_rank now {new_total}", flush=True)
