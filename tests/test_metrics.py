import numpy as np
import pytest

from brats_pipeline.inference import (
    brats_region_dice_np,
    brats_region_voxels,
    dice_binary_np,
    make_metric_row,
    make_pred_with_et_threshold,
)


def test_binary_dice_identical_and_empty_masks_are_one():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert dice_binary_np(mask, mask) == pytest.approx(1.0)

    empty = np.zeros((4, 4), dtype=np.uint8)
    assert dice_binary_np(empty, empty) == pytest.approx(1.0)


def test_binary_dice_known_overlap():
    pred = np.array([1, 1, 0, 0], dtype=np.uint8)
    target = np.array([1, 0, 1, 0], dtype=np.uint8)

    # intersection=1, |pred|=2, |target|=2 -> Dice=0.5
    assert dice_binary_np(pred, target) == pytest.approx(0.5, abs=1e-6)


def test_brats_region_dice_uses_nested_wt_tc_et_regions():
    target = np.array([0, 1, 2, 3], dtype=np.uint8).reshape(1, 2, 2)
    pred = np.array([0, 1, 0, 3], dtype=np.uint8).reshape(1, 2, 2)

    metrics = brats_region_dice_np(pred, target)

    assert metrics["dice_WT"] == pytest.approx(0.8, abs=1e-6)
    assert metrics["dice_TC"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["dice_ET"] == pytest.approx(1.0, abs=1e-6)


def test_region_voxels_accept_raw_label_4():
    seg = np.array([0, 1, 2, 4], dtype=np.uint8).reshape(1, 2, 2)
    voxels = brats_region_voxels(seg)

    assert voxels == {
        "WT_vox": 3,
        "TC_vox": 2,
        "ET_vox": 1,
    }


def test_et_probability_threshold_changes_only_et_assignment_rule():
    # [C, D, H, W] = [4, 1, 1, 3]
    probs = np.zeros((4, 1, 1, 3), dtype=np.float32)

    # Voxel 0: class 0 is the best non-ET class; ET below threshold.
    probs[:, 0, 0, 0] = [0.70, 0.10, 0.10, 0.65]
    # Voxel 1: class 1 is the best non-ET class; ET above threshold.
    probs[:, 0, 0, 1] = [0.05, 0.55, 0.10, 0.90]
    # Voxel 2: class 2 is the best non-ET class; ET below threshold.
    probs[:, 0, 0, 2] = [0.10, 0.20, 0.60, 0.79]

    pred = make_pred_with_et_threshold(probs, et_thr=0.80)
    np.testing.assert_array_equal(pred.ravel(), np.array([0, 3, 2], dtype=np.uint8))


def test_et_threshold_rejects_invalid_inputs():
    probs = np.zeros((4, 2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_pred_with_et_threshold(probs, -0.01)

    with pytest.raises(ValueError, match=r"\[4, D, H, W\]"):
        make_pred_with_et_threshold(np.zeros((3, 2, 2, 2), dtype=np.float32), 0.5)


def test_make_metric_row_reports_consistent_dice_volumes_and_ratios():
    target = np.array([0, 1, 2, 3], dtype=np.uint8).reshape(1, 2, 2)
    pred = np.array([0, 1, 0, 3], dtype=np.uint8).reshape(1, 2, 2)

    row = make_metric_row("toy", pred, target, et_threshold=0.95)

    assert row["model"] == "toy"
    assert row["dice_WT"] == pytest.approx(0.8, abs=1e-6)
    assert row["dice_TC"] == pytest.approx(1.0, abs=1e-6)
    assert row["dice_ET"] == pytest.approx(1.0, abs=1e-6)
    assert row["mean_dice"] == pytest.approx((0.8 + 1.0 + 1.0) / 3.0, abs=1e-6)

    assert row["gt_WT_vox"] == 3
    assert row["pred_WT_vox"] == 2
    assert row["WT_ratio"] == pytest.approx(2 / 3)
    assert row["TC_ratio"] == pytest.approx(1.0)
    assert row["ET_ratio"] == pytest.approx(1.0)
    assert row["et_threshold"] == pytest.approx(0.95)
