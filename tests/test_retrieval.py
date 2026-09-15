from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from brats_pipeline.retrieval import (
    BaselineContext,
    bbox_sizes,
    compactness_proxy,
    compute_neighbors,
    compute_tumor_features,
    ensure_no_nan_matrix,
    l2_normalize_np,
    pool_feature_map,
)


def test_l2_normalize_rows_and_preserve_zero_row():
    x = np.array(
        [
            [3.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    out = l2_normalize_np(x)

    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    np.testing.assert_array_equal(out[1], np.array([0.0, 0.0], dtype=np.float32))


def test_nonfinite_embedding_guard():
    ensure_no_nan_matrix("clean", np.eye(3, dtype=np.float32))

    with pytest.raises(ValueError, match="NaN/Inf"):
        ensure_no_nan_matrix("bad", np.array([[1.0, np.nan]], dtype=np.float32))


def test_bbox_sizes_and_compactness_proxy():
    mask = np.zeros((6, 7, 8), dtype=bool)
    mask[1:4, 2:6, 3:5] = True

    assert bbox_sizes(mask) == (3, 4, 2)
    # The mask completely fills its bounding box.
    assert compactness_proxy(mask) == pytest.approx(1.0)

    empty = np.zeros_like(mask)
    assert bbox_sizes(empty) == (0, 0, 0)
    assert compactness_proxy(empty) == pytest.approx(0.0)


def test_compute_neighbors_excludes_self_and_orders_by_cosine_similarity():
    x = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    x = l2_normalize_np(x)

    indices, similarities = compute_neighbors(x, max_k=2)

    assert indices.shape == (3, 2)
    assert similarities.shape == (3, 2)
    for query_idx in range(3):
        assert query_idx not in indices[query_idx]

    # Patient 0 should retrieve patient 1 before patient 2.
    assert indices[0, 0] == 1
    assert similarities[0, 0] > similarities[0, 1]


def test_pool_feature_map_gap_gmp_shape_and_values():
    feat = torch.tensor(
        [
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[5.0, 6.0], [7.0, 8.0]]],
            ]
        ]
    )
    # Shape is [B=1, C=2, D=1, H=2, W=2].
    out = pool_feature_map(feat, mode="gap_gmp")

    assert out.shape == (1, 4)
    torch.testing.assert_close(out, torch.tensor([[2.5, 6.5, 4.0, 8.0]]))

    with pytest.raises(ValueError, match="gap, gmp, or gap_gmp"):
        pool_feature_map(feat, mode="bad")


def test_compute_tumor_features_region_counts_and_ratios():
    image = np.zeros((4, 5, 5, 5), dtype=np.float32)
    image[:, 1:4, 1:4, 1:4] = 1.0

    seg = np.zeros((5, 5, 5), dtype=np.uint8)
    seg[1:4, 1:4, 1:4] = 2
    seg[2:4, 2:4, 2:4] = 1
    seg[3, 3, 3] = 3

    case = {
        "image_xyz": image,
        "seg_xyz": seg,
        "patient_id": "toy",
        "client_domain": "domain_a",
        "split": "test",
    }

    features = compute_tumor_features(case)

    assert features["patient_id"] == "toy"
    assert features["WT_vox"] == int((seg > 0).sum())
    assert features["TC_vox"] == int(((seg == 1) | (seg == 3)).sum())
    assert features["ET_vox"] == 1
    assert features["ET_present"] is True
    assert features["ET_to_TC_ratio"] == pytest.approx(features["ET_vox"] / features["TC_vox"])
    assert 0.0 < features["WT_fraction"] <= 1.0


def _make_baseline_context():
    patient_ids = ["p0", "p1", "p2", "p3"]
    domain_arr = np.array(["A", "A", "B", "B"])
    et_present = np.array([True, True, False, False])
    log_wt = np.array([1.0, 1.2, 2.0, 2.2])
    log_tc = np.array([0.5, 0.7, 1.5, 1.7])
    log_et = np.array([0.2, 0.3, 0.0, 0.0])
    centroids = {
        "WT": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        "TC": np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [5.0, 1.0, 0.0], [6.0, 1.0, 0.0]]),
        "ET": np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
    }
    tumor_feature_z = np.eye(4, dtype=np.float32)
    expected_same = np.array([1 / 3, 1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

    # One observed neighbor per query, deliberately alternating domains.
    neighbors_df = pd.DataFrame(
        [
            {"experiment_key": "exp", "retrieval_space": "embedding", "query_patient_id": "p0", "neighbor_rank": 1, "neighbor_domain": "B"},
            {"experiment_key": "exp", "retrieval_space": "embedding", "query_patient_id": "p1", "neighbor_rank": 1, "neighbor_domain": "B"},
            {"experiment_key": "exp", "retrieval_space": "embedding", "query_patient_id": "p2", "neighbor_rank": 1, "neighbor_domain": "A"},
            {"experiment_key": "exp", "retrieval_space": "embedding", "query_patient_id": "p3", "neighbor_rank": 1, "neighbor_domain": "A"},
        ]
    )

    metric_cols = [
        "same_domain_rate",
        "cross_domain_rate",
        "expected_same_domain_rate",
        "same_domain_enrichment",
        "same_ET_presence_rate",
        "mean_abs_log1p_WT_diff",
        "mean_abs_log1p_TC_diff",
        "mean_abs_log1p_ET_diff",
        "mean_tumor_feature_z_distance",
        "mean_WT_centroid_distance",
        "mean_TC_centroid_distance",
        "mean_ET_centroid_distance",
    ]

    return BaselineContext(
        patient_ids=patient_ids,
        domain_arr=domain_arr,
        ET_present=et_present,
        log1p_WT=log_wt,
        log1p_TC=log_tc,
        log1p_ET=log_et,
        centroids=centroids,
        tumor_feature_z=tumor_feature_z,
        expected_same_by_idx=expected_same,
        neighbors_df=neighbors_df,
        metric_cols=metric_cols,
    )


def test_baseline_global_random_never_returns_self_neighbor():
    context = _make_baseline_context()
    rng = np.random.default_rng(42)

    sampled = context.sample_global_random(k=2, rng=rng)

    for query_idx, neighbors in sampled.items():
        assert len(neighbors) == 2
        assert query_idx not in neighbors
        assert len(set(neighbors.tolist())) == 2


def test_baseline_same_domain_strict_uses_only_same_domain_candidates():
    context = _make_baseline_context()
    rng = np.random.default_rng(42)

    sampled = context.sample_same_domain_strict(k=3, rng=rng)

    for query_idx, neighbors in sampled.items():
        # Each domain has only two patients, so one eligible same-domain neighbor remains.
        assert len(neighbors) == 1
        assert context.domain_arr[neighbors[0]] == context.domain_arr[query_idx]
        assert neighbors[0] != query_idx


def test_baseline_evaluation_returns_expected_query_and_pair_counts():
    context = _make_baseline_context()
    sampled = {
        0: np.array([1], dtype=int),
        1: np.array([0], dtype=int),
        2: np.array([3], dtype=int),
        3: np.array([2], dtype=int),
    }

    result = context.evaluate_sampled_neighbors(sampled)

    assert result["n_queries"] == 4
    assert result["n_pairs"] == 4
    assert result["effective_k_mean"] == pytest.approx(1.0)
    assert result["same_domain_rate"] == pytest.approx(1.0)
    assert result["cross_domain_rate"] == pytest.approx(0.0)
