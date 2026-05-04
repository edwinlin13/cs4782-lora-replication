import torch
from torch import nn
from transformers import GPT2LMHeadModel
from sequential_lora import inject_sequential_lora, append_stage_to_model, LoRAQVStack
from sequential_train import FixedStepTrigger, PlateauTrigger, SequentialOptimizerManager


PER_STAGE_RANK = 2
PER_STAGE_ALPHA = 2
BASE_LR = 2e-4


def _make_seq_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inject_sequential_lora(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    return model


# ---- FixedStepTrigger ----

def test_fixed_step_fires_at_each_boundary():
    """4 stages over 1000 steps -> fire at 250, 500, 750."""
    trig = FixedStepTrigger(total_steps=1000, num_stages=4)
    fires = [step for step in range(1, 1001) if trig.update(step=step).get("stage")]
    assert fires == [250, 500, 750], f"got {fires}"


def test_fixed_step_no_fire_at_step_zero_or_total():
    """Don't fire at step 0 (we start with stage 1) or at the final step."""
    trig = FixedStepTrigger(total_steps=1000, num_stages=4)
    assert trig.update(step=0) == {}
    assert trig.update(step=1000) == {}


def test_fixed_step_one_stage_never_fires():
    """1 stage means no transitions at all."""
    trig = FixedStepTrigger(total_steps=1000, num_stages=1)
    fires = [step for step in range(1, 1001) if trig.update(step=step).get("stage")]
    assert fires == []


def test_fixed_step_idempotent_on_repeat_call():
    """Calling update twice on the same boundary step must only fire once.
    The driver pings the trigger after the train step and again after eval
    (when eval_every aligns w/ a boundary), so this matters."""
    trig = FixedStepTrigger(total_steps=8, num_stages=2)  # boundary at 4
    first = trig.update(step=4)
    second = trig.update(step=4)
    assert first.get("stage")
    assert not second.get("stage")


# ---- PlateauTrigger ----

def test_plateau_does_not_fire_when_improving():
    """Strictly decreasing val_loss should never trigger."""
    trig = PlateauTrigger(patience=3, delta=0.001, max_total_rank=10, per_stage_rank=2)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    decisions = [trig.update(val_loss=l, current_total_rank=2) for l in losses]
    assert not any(d.get("stage") for d in decisions)
    assert not any(d.get("stop") for d in decisions)


def test_plateau_fires_after_patience_no_improvement():
    """Flat val_loss for `patience` evals -> trigger stage."""
    trig = PlateauTrigger(patience=3, delta=0.001, max_total_rank=10, per_stage_rank=2)
    # one improvement, then 3 flat
    decisions = []
    for l in [1.0, 0.9, 0.9, 0.9, 0.9]:
        decisions.append(trig.update(val_loss=l, current_total_rank=2))
    # the 4th one should fire (3 evals at 0.9 with no improvement vs the prior 0.9)
    fires = [i for i, d in enumerate(decisions) if d.get("stage")]
    assert len(fires) == 1, f"expected 1 fire, got {fires}"


def test_plateau_signals_stop_after_post_stage_no_improvement():
    """Once a stage fires, if patience more evals show no improvement, stop."""
    trig = PlateauTrigger(patience=2, delta=0.001, max_total_rank=10, per_stage_rank=2)
    # decreasing then flat to fire stage
    trig.update(val_loss=1.0, current_total_rank=2)
    trig.update(val_loss=0.9, current_total_rank=2)
    trig.update(val_loss=0.9, current_total_rank=2)
    d = trig.update(val_loss=0.9, current_total_rank=2)
    assert d.get("stage"), "expected a stage trigger here"
    # now post-stage at total_rank=4: patience=2 flat evals -> stop
    d1 = trig.update(val_loss=0.9, current_total_rank=4)
    d2 = trig.update(val_loss=0.9, current_total_rank=4)
    assert d2.get("stop"), "expected stop signal after post-stage plateau"


def test_plateau_caps_total_rank():
    """Once total_rank >= max_total_rank, never fire again -- signal stop instead."""
    trig = PlateauTrigger(patience=2, delta=0.001, max_total_rank=4, per_stage_rank=2)
    # already at cap (rank=4), so a plateau means stop, not stage
    trig.update(val_loss=0.5, current_total_rank=4)
    trig.update(val_loss=0.5, current_total_rank=4)
    d = trig.update(val_loss=0.5, current_total_rank=4)
    assert d.get("stop")
    assert not d.get("stage")


def test_plateau_no_op_on_per_step_call_without_val_loss():
    """The driver pings every trigger on each train step (no val_loss). Plateau
    must no-op gracefully, not crash on missing val_loss."""
    trig = PlateauTrigger(patience=3, delta=0.001, max_total_rank=10, per_stage_rank=2)
    # per-step call style
    d = trig.update(step=42, current_total_rank=2)
    assert d == {}, f"expected no-op, got {d}"


def test_optim_manager_initial_active_group_at_base_lr():
    model = _make_seq_model()
    mgr = SequentialOptimizerManager(model, base_lr=BASE_LR, alpha_old=0.0,
                                     weight_decay=0.01, warmup_steps=500)
    # one group, all current-stage params, initial_lr = base_lr
    assert len(mgr.optimizer.param_groups) == 1
    assert mgr.optimizer.param_groups[0]["initial_lr"] == BASE_LR


def test_optim_manager_alpha_zero_freezes_old_stages():
    model = _make_seq_model()
    mgr = SequentialOptimizerManager(model, base_lr=BASE_LR, alpha_old=0.0,
                                     weight_decay=0.01, warmup_steps=500)
    new_stages = append_stage_to_model(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    mgr.on_stage_added(new_stages)

    # the previously-active stages must now have requires_grad=False
    for _, m in model.named_modules():
        if isinstance(m, LoRAQVStack):
            for p in m.stages[0].parameters():  # the original first stage
                assert not p.requires_grad
            for p in m.stages[1].parameters():  # the freshly added one
                assert p.requires_grad

    # optimizer should have one group, with only the new stage's params
    assert len(mgr.optimizer.param_groups) == 1


def test_optim_manager_alpha_nonzero_keeps_two_groups_with_correct_ratio():
    model = _make_seq_model()
    mgr = SequentialOptimizerManager(model, base_lr=BASE_LR, alpha_old=0.1,
                                     weight_decay=0.01, warmup_steps=500)
    new_stages = append_stage_to_model(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    mgr.on_stage_added(new_stages)

    assert len(mgr.optimizer.param_groups) == 2
    initial_lrs = [g["initial_lr"] for g in mgr.optimizer.param_groups]
    # one group at base_lr (active), one at alpha_old * base_lr (old)
    assert BASE_LR in initial_lrs
    assert 0.1 * BASE_LR in [round(lr, 12) for lr in initial_lrs]


def test_optim_manager_alpha_one_keeps_old_at_full_lr():
    model = _make_seq_model()
    mgr = SequentialOptimizerManager(model, base_lr=BASE_LR, alpha_old=1.0,
                                     weight_decay=0.01, warmup_steps=500)
    new_stages = append_stage_to_model(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    mgr.on_stage_added(new_stages)
    # both groups at base_lr (since alpha_old=1)
    for g in mgr.optimizer.param_groups:
        assert g["initial_lr"] == BASE_LR


def test_optim_manager_warmup_scaling():
    """During warmup, lr = initial_lr * warmup_factor."""
    model = _make_seq_model()
    mgr = SequentialOptimizerManager(model, base_lr=BASE_LR, alpha_old=0.5,
                                     weight_decay=0.01, warmup_steps=500)
    # at step 0, factor is essentially 0 -> lr near 0
    mgr.set_step(0)
    for g in mgr.optimizer.param_groups:
        assert g["lr"] < BASE_LR
    # at step 500+, factor is 1 -> lr = initial_lr
    mgr.set_step(500)
    for g in mgr.optimizer.param_groups:
        assert g["lr"] == g["initial_lr"]
    mgr.set_step(10000)
    for g in mgr.optimizer.param_groups:
        assert g["lr"] == g["initial_lr"]


def test_optim_manager_preserves_old_param_state_across_stage():
    """Adam moments (exp_avg, exp_avg_sq) must NOT be reset for old params."""
    model = _make_seq_model()
    mgr = SequentialOptimizerManager(model, base_lr=BASE_LR, alpha_old=1.0,
                                     weight_decay=0.01, warmup_steps=0)
    mgr.set_step(0)

    # take a step so optimizer state is populated for the original stage
    x = torch.randint(0, 50257, (1, 8))
    out = model(x, labels=x)
    out.loss.backward()
    mgr.optimizer.step()
    mgr.optimizer.zero_grad()

    # snapshot old stage's exp_avg
    # use lora_q_B because A has zero gradient on step 1 (since dL/dA proportional to B=0 init)
    # so we'd snapshot a zero exp_avg and the "state preserved" check would be vacuous
    original_stage_param = next(
        m.stages[0].lora_q_B
        for _, m in model.named_modules() if isinstance(m, LoRAQVStack)
    )
    snapshot = mgr.optimizer.state[original_stage_param]["exp_avg"].clone()
    assert snapshot.abs().sum() > 0  # sanity: state exists

    # add a stage
    new_stages = append_stage_to_model(model, rank=PER_STAGE_RANK, alpha=PER_STAGE_ALPHA)
    mgr.on_stage_added(new_stages)

    # old param's state must still be present and unchanged
    after = mgr.optimizer.state[original_stage_param]["exp_avg"]
    assert torch.allclose(after, snapshot)


if __name__ == "__main__":
    test_fixed_step_fires_at_each_boundary()
    test_fixed_step_no_fire_at_step_zero_or_total()
    test_fixed_step_one_stage_never_fires()
    test_fixed_step_idempotent_on_repeat_call()
    test_plateau_does_not_fire_when_improving()
    test_plateau_fires_after_patience_no_improvement()
    test_plateau_signals_stop_after_post_stage_no_improvement()
    test_plateau_caps_total_rank()
    test_plateau_no_op_on_per_step_call_without_val_loss()
    print("Trigger tests passed!")
    print("Running optimizer manager tests (downloads gpt2)...")
    test_optim_manager_initial_active_group_at_base_lr()
    test_optim_manager_alpha_zero_freezes_old_stages()
    test_optim_manager_alpha_nonzero_keeps_two_groups_with_correct_ratio()
    test_optim_manager_alpha_one_keeps_old_at_full_lr()
    test_optim_manager_warmup_scaling()
    test_optim_manager_preserves_old_param_state_across_stage()
    print("Optimizer manager tests passed!")
