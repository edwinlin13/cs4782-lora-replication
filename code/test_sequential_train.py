from sequential_train import FixedStepTrigger, PlateauTrigger


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


if __name__ == "__main__":
    test_fixed_step_fires_at_each_boundary()
    test_fixed_step_no_fire_at_step_zero_or_total()
    test_fixed_step_one_stage_never_fires()
    test_plateau_does_not_fire_when_improving()
    test_plateau_fires_after_patience_no_improvement()
    test_plateau_signals_stop_after_post_stage_no_improvement()
    test_plateau_caps_total_rank()
    print("Trigger tests passed!")
