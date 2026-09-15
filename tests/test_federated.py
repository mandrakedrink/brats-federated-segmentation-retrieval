import copy

import numpy as np
import pytest
import torch

from brats_pipeline.training import (
    WeightedDiceCELoss,
    brats_region_dice,
    fedavg_state_dicts,
    fedprox_l2_penalty,
    get_cpu_state_dict,
    get_round_lr,
    make_global_reference_state,
)


def test_round_lr_starts_at_base_ends_at_min_and_is_monotonic():
    total_rounds = 10
    values = [get_round_lr(i, total_rounds, base_lr=1e-4, min_lr=1e-6) for i in range(1, total_rounds + 1)]

    assert values[0] == pytest.approx(1e-4)
    assert values[-1] == pytest.approx(1e-6)
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_fedavg_weighted_average_and_integer_buffer_policy():
    states = {
        "client_a": {
            "weight": torch.tensor([1.0, 3.0]),
            "num_batches_tracked": torch.tensor(7, dtype=torch.long),
        },
        "client_b": {
            "weight": torch.tensor([5.0, 7.0]),
            "num_batches_tracked": torch.tensor(99, dtype=torch.long),
        },
    }
    weights = {"client_a": 0.25, "client_b": 0.75}

    averaged = fedavg_state_dicts(states, weights)

    torch.testing.assert_close(averaged["weight"], torch.tensor([4.0, 6.0]))
    # Non-floating buffers follow the first client by design.
    assert averaged["num_batches_tracked"].item() == 7


def test_get_cpu_state_dict_returns_detached_clones():
    model = torch.nn.Linear(3, 2)
    copied = get_cpu_state_dict(model)

    original_before = copied["weight"].clone()
    with torch.no_grad():
        model.weight.add_(10.0)

    assert copied["weight"].device.type == "cpu"
    assert copied["weight"].requires_grad is False
    torch.testing.assert_close(copied["weight"], original_before)
    assert not torch.allclose(copied["weight"], model.weight.detach().cpu())


def test_global_reference_contains_only_floating_tensors():
    state = {
        "w": torch.tensor([1.0, 2.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }

    ref = make_global_reference_state(state, "cpu")

    assert set(ref) == {"w"}
    torch.testing.assert_close(ref["w"], state["w"])


def test_fedprox_penalty_is_zero_at_reference_and_positive_after_change():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))

    reference = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert fedprox_l2_penalty(model, reference).item() == pytest.approx(0.0)

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 4.0]]))

    # (2-1)^2 + (4-2)^2 = 5
    assert fedprox_l2_penalty(model, reference).item() == pytest.approx(5.0)


def test_torch_brats_region_dice_matches_expected_nested_regions():
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long).reshape(1, 2, 2)
    pred = torch.tensor([0, 1, 0, 3], dtype=torch.long).reshape(1, 2, 2)

    metrics = brats_region_dice(pred, target)

    assert metrics["WT"] == pytest.approx(0.8, abs=1e-6)
    assert metrics["TC"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["ET"] == pytest.approx(1.0, abs=1e-6)


def test_weighted_dice_ce_loss_forward_backward_if_monai_is_available():
    pytest.importorskip("monai")

    loss_fn = WeightedDiceCELoss(
        ce_weight=[0.2, 1.0, 1.0, 1.25],
        dice_weight=1.0,
        ce_loss_weight=1.0,
    )

    logits = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 4, (2, 4, 4, 4), dtype=torch.long)

    loss = loss_fn(logits, target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
