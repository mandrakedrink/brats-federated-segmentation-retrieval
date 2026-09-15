import numpy as np
import pandas as pd
import pytest

from brats_pipeline.data import (
    best_tumor_slice_dhw,
    canonicalize_brats_labels,
    crop_patch_xyz,
    crop_patch_xyz_retrieval,
    dhw_to_xyz,
    image_foreground_center_xyz,
    normalize_nonzero,
    remap_brats_labels,
    select_manifest_row,
    tumor_center_xyz,
    tumor_center_xyz_retrieval,
    validate_label_values,
    xyz_to_dhw,
    xyz_to_dhw_seg,
)


def test_normalize_nonzero_preserves_background_and_standardizes_positive_voxels():
    x = np.array(
        [
            [[0.0, 1.0], [2.0, -5.0]],
            [[3.0, 4.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    out = normalize_nonzero(x)
    positive = x > 0

    assert out.dtype == np.float32
    assert np.all(out[~positive] == 0.0)
    assert np.isclose(out[positive].mean(), 0.0, atol=1e-6)
    assert np.isclose(out[positive].std(), 1.0, atol=1e-6)


def test_normalize_nonzero_all_background_returns_zeros():
    x = np.zeros((3, 4, 5), dtype=np.float32)
    out = normalize_nonzero(x)
    np.testing.assert_array_equal(out, x)


def test_brats_label_mapping_and_canonicalization():
    raw = np.array([0, 1, 2, 4], dtype=np.uint8)
    remapped = remap_brats_labels(raw)
    np.testing.assert_array_equal(remapped, np.array([0, 1, 2, 3], dtype=np.uint8))

    already_contiguous = np.array([0, 1, 2, 3], dtype=np.uint8)
    canonical = canonicalize_brats_labels(already_contiguous)
    np.testing.assert_array_equal(canonical, already_contiguous)


def test_xyz_dhw_roundtrip_for_segmentation():
    seg_xyz = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    seg_dhw = xyz_to_dhw_seg(seg_xyz)

    assert seg_dhw.shape == (4, 3, 2)
    np.testing.assert_array_equal(dhw_to_xyz(seg_dhw), seg_xyz)


def test_xyz_to_dhw_preserves_channel_axis_and_transposes_spatial_axes():
    image_xyz = np.arange(4 * 2 * 3 * 5, dtype=np.float32).reshape(4, 2, 3, 5)
    seg_xyz = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 3, 5)

    image_dhw, seg_dhw = xyz_to_dhw(image_xyz, seg_xyz)

    assert image_dhw.shape == (4, 5, 3, 2)
    assert seg_dhw.shape == (5, 3, 2)
    np.testing.assert_array_equal(image_dhw[:, 0, :, :], image_xyz[:, :, :, 0].transpose(0, 2, 1))
    np.testing.assert_array_equal(dhw_to_xyz(seg_dhw), seg_xyz)


def test_training_crop_zero_pads_at_volume_boundary():
    image = np.ones((4, 5, 5, 5), dtype=np.float32)
    seg = np.ones((5, 5, 5), dtype=np.uint8)

    image_patch, seg_patch = crop_patch_xyz(
        image,
        seg,
        center_xyz=(0, 0, 0),
        patch_size=(4, 4, 4),
    )

    assert image_patch.shape == (4, 4, 4, 4)
    assert seg_patch.shape == (4, 4, 4)
    # With center at the corner, the leading half of each axis is padding.
    assert np.all(image_patch[:, :2, :, :] == 0)
    assert np.all(image_patch[:, :, :2, :] == 0)
    assert np.all(image_patch[:, :, :, :2] == 0)
    assert np.all(seg_patch[:2, :, :] == 0)
    assert image_patch[:, 2:, 2:, 2:].sum() > 0


def test_retrieval_crop_keeps_requested_shape_for_fractional_center():
    image = np.ones((4, 7, 8, 9), dtype=np.float32)
    seg = np.ones((7, 8, 9), dtype=np.uint8)

    image_patch, seg_patch = crop_patch_xyz_retrieval(
        image,
        seg,
        center_xyz=(3.5, 4.0, 4.5),
        patch_size=(5, 5, 5),
    )

    assert image_patch.shape == (4, 5, 5, 5)
    assert seg_patch.shape == (5, 5, 5)


def test_tumor_center_and_empty_fallback():
    seg = np.zeros((9, 11, 13), dtype=np.uint8)
    assert tumor_center_xyz(seg) == (4, 5, 6)

    seg[2:4, 6:8, 8:10] = 2
    # Mean coordinates are (2.5, 6.5, 8.5); the training helper truncates.
    assert tumor_center_xyz(seg) == (2, 6, 8)
    # Retrieval helper intentionally rounds its ROI center.
    assert tumor_center_xyz_retrieval(seg, region="WT") == (2, 6, 8)


def test_retrieval_region_center_supports_wt_tc_et():
    seg = np.zeros((7, 7, 7), dtype=np.uint8)
    seg[1:6, 1:6, 1:6] = 2
    seg[2:5, 2:5, 2:5] = 1
    seg[3, 3, 3] = 3

    assert tumor_center_xyz_retrieval(seg, "WT") == (3, 3, 3)
    assert tumor_center_xyz_retrieval(seg, "TC") == (3, 3, 3)
    assert tumor_center_xyz_retrieval(seg, "ET") == (3, 3, 3)

    with pytest.raises(ValueError, match="WT, TC, or ET"):
        tumor_center_xyz_retrieval(seg, "invalid")


def test_image_foreground_center_has_volume_center_fallback():
    image = np.zeros((4, 8, 10, 12), dtype=np.float32)
    assert image_foreground_center_xyz(image) == (4, 5, 6)

    image[:, 2:4, 6:8, 8:10] = 1.0
    assert image_foreground_center_xyz(image) == (2, 6, 8)


def test_best_tumor_slice_and_empty_fallback():
    seg = np.zeros((9, 6, 6), dtype=np.uint8)
    assert best_tumor_slice_dhw(seg) == 4

    seg[2, :2, :2] = 1
    seg[7, :4, :4] = 2
    assert best_tumor_slice_dhw(seg) == 7


def test_label_validation_and_manifest_selection():
    validate_label_values(np.array([0, 1, 2, 4]), allow_raw=True)
    validate_label_values(np.array([0, 1, 2, 3]), allow_raw=False)

    with pytest.raises(ValueError, match="Unexpected label values"):
        validate_label_values(np.array([0, 1, 7]), allow_raw=True)

    manifest = pd.DataFrame(
        {
            "patient_id": ["BraTS2021_00001", "BraTS2021_00002"],
            "split": ["train", "test"],
        }
    )

    row = select_manifest_row(manifest, "BraTS2021_00002")
    assert row["split"] == "test"

    with pytest.raises(ValueError, match="Expected one manifest row"):
        select_manifest_row(manifest, "missing")
