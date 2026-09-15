"""Encoder features, phenotype definitions and random retrieval controls.

The GT-centered and predicted-ROI notebooks use different phenotype schemas.
Both are kept here without treating their distances as interchangeable.
"""
from pathlib import Path
from collections import Counter
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from .data import xyz_to_dhw, mask_center_xyz
from .inference import create_segresnet, load_trusted_checkpoint, extract_model_state_dict

def l2_normalize_np(x, eps=1e-12):
    """
    L2 normalize matrix row-wise.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)

def ensure_no_nan_matrix(name, x):
    """Reject a matrix containing non-finite values."""
    if not np.isfinite(x).all():
        bad = np.where(~np.isfinite(x))
        raise ValueError(f'{name} contains NaN/Inf at indices {bad[:5]}')

def mask_stats(seg_xyz, label_mask, prefix):
    """
    Volume, centroid, bbox and compactness proxy for a binary mask.
    """
    mask = label_mask.astype(bool)
    shape = np.array(seg_xyz.shape, dtype=np.float32)
    vol = int(mask.sum())
    out = {f'{prefix}_vox': vol, f'{prefix}_present': bool(vol > 0)}
    if vol == 0:
        out.update({
            f'{prefix}_centroid_x': np.nan,
            f'{prefix}_centroid_y': np.nan,
            f'{prefix}_centroid_z': np.nan,
            f'{prefix}_bbox_x': 0.0,
            f'{prefix}_bbox_y': 0.0,
            f'{prefix}_bbox_z': 0.0,
            f'{prefix}_bbox_vox': 0,
            f'{prefix}_compactness': 0.0
        })
        return out
    pts = np.argwhere(mask).astype(np.float32)
    centroid = pts.mean(axis=0) / np.maximum(shape - 1, 1)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    bbox_dims = (maxs - mins + 1).astype(np.float32)
    bbox_norm = bbox_dims / np.maximum(shape, 1)
    bbox_vox = int(np.prod(bbox_dims))
    compactness = float(vol / max(bbox_vox, 1))
    out.update({
        f'{prefix}_centroid_x': float(centroid[0]),
        f'{prefix}_centroid_y': float(centroid[1]),
        f'{prefix}_centroid_z': float(centroid[2]),
        f'{prefix}_bbox_x': float(bbox_norm[0]),
        f'{prefix}_bbox_y': float(bbox_norm[1]),
        f'{prefix}_bbox_z': float(bbox_norm[2]),
        f'{prefix}_bbox_vox': bbox_vox,
        f'{prefix}_compactness': compactness
    })
    return out

def compute_tumor_phenotype_features(seg_xyz):
    """
    Computes WT/TC/ET phenotype features.

    Remapped labels:
    - WT = labels 1,2,3
    - TC = labels 1,3
    - ET = label 3
    """
    wt = seg_xyz > 0
    tc = (seg_xyz == 1) | (seg_xyz == 3)
    et = seg_xyz == 3
    ed = seg_xyz == 2
    ncr = seg_xyz == 1
    total_vox = int(np.prod(seg_xyz.shape))
    eps = 1e-08
    out = {
        'volume_total_vox': total_vox,
        'shape_x': int(seg_xyz.shape[0]),
        'shape_y': int(seg_xyz.shape[1]),
        'shape_z': int(seg_xyz.shape[2])
    }
    out.update(mask_stats(seg_xyz, wt, 'WT'))
    out.update(mask_stats(seg_xyz, tc, 'TC'))
    out.update(mask_stats(seg_xyz, et, 'ET'))
    out.update(mask_stats(seg_xyz, ed, 'ED'))
    out.update(mask_stats(seg_xyz, ncr, 'NCR'))
    for region in ['WT', 'TC', 'ET', 'ED', 'NCR']:
        out[f'{region}_fraction'] = out[f'{region}_vox'] / max(total_vox, 1)
        out[f'log1p_{region}_vox'] = float(np.log1p(out[f'{region}_vox']))
    out['TC_to_WT_ratio'] = out['TC_vox'] / (out['WT_vox'] + eps)
    out['ET_to_WT_ratio'] = out['ET_vox'] / (out['WT_vox'] + eps)
    out['ET_to_TC_ratio'] = out['ET_vox'] / (out['TC_vox'] + eps)
    out['ED_to_WT_ratio'] = out['ED_vox'] / (out['WT_vox'] + eps)
    return out

def bbox_sizes(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return (0, 0, 0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return tuple((maxs - mins + 1).astype(int).tolist())

def compactness_proxy(mask):
    vox = int(mask.sum())
    if vox == 0:
        return 0.0
    bx, by, bz = bbox_sizes(mask)
    bbox_vol = max(int(bx * by * bz), 1)
    return float(vox / bbox_vol)

def centroid_or_nan(mask):
    c = mask_center_xyz(mask)
    if c is None:
        return (np.nan, np.nan, np.nan)
    return tuple((float(v) for v in c))

def compute_tumor_features(case):
    seg = case['seg_xyz']
    WT = seg > 0
    TC = (seg == 1) | (seg == 3)
    ET = seg == 3
    ED = seg == 2
    wt_vox = int(WT.sum())
    tc_vox = int(TC.sum())
    et_vox = int(ET.sum())
    ed_vox = int(ED.sum())
    brain_fg = np.any(case['image_xyz'] != 0, axis=0)
    brain_vox = max(int(brain_fg.sum()), 1)
    wt_c = centroid_or_nan(WT)
    tc_c = centroid_or_nan(TC)
    et_c = centroid_or_nan(ET)
    wt_bbox = bbox_sizes(WT)
    tc_bbox = bbox_sizes(TC)
    et_bbox = bbox_sizes(ET)
    return {
        'patient_id': case['patient_id'],
        'client_domain': case['client_domain'],
        'split': case['split'],
        'WT_vox': wt_vox,
        'TC_vox': tc_vox,
        'ET_vox': et_vox,
        'ED_vox': ed_vox,
        'log1p_WT_vox': np.log1p(wt_vox),
        'log1p_TC_vox': np.log1p(tc_vox),
        'log1p_ET_vox': np.log1p(et_vox),
        'log1p_ED_vox': np.log1p(ed_vox),
        'WT_fraction': wt_vox / brain_vox,
        'TC_fraction': tc_vox / brain_vox,
        'ET_fraction': et_vox / brain_vox,
        'ED_fraction': ed_vox / brain_vox,
        'TC_to_WT_ratio': tc_vox / max(wt_vox, 1),
        'ET_to_WT_ratio': et_vox / max(wt_vox, 1),
        'ET_to_TC_ratio': et_vox / max(tc_vox, 1),
        'ED_to_WT_ratio': ed_vox / max(wt_vox, 1),
        'WT_centroid_x': wt_c[0],
        'WT_centroid_y': wt_c[1],
        'WT_centroid_z': wt_c[2],
        'TC_centroid_x': tc_c[0],
        'TC_centroid_y': tc_c[1],
        'TC_centroid_z': tc_c[2],
        'ET_centroid_x': et_c[0],
        'ET_centroid_y': et_c[1],
        'ET_centroid_z': et_c[2],
        'WT_bbox_x': wt_bbox[0],
        'WT_bbox_y': wt_bbox[1],
        'WT_bbox_z': wt_bbox[2],
        'TC_bbox_x': tc_bbox[0],
        'TC_bbox_y': tc_bbox[1],
        'TC_bbox_z': tc_bbox[2],
        'ET_bbox_x': et_bbox[0],
        'ET_bbox_y': et_bbox[1],
        'ET_bbox_z': et_bbox[2],
        'WT_compactness': compactness_proxy(WT),
        'TC_compactness': compactness_proxy(TC),
        'ET_compactness': compactness_proxy(ET),
        'ET_present': bool(et_vox > 0)
    }

def encode_segresnet_features(model, x):
    """
    Returns encoder feature maps from SegResNet.
    """
    if hasattr(model, 'encode'):
        encoded = model.encode(x)
        if isinstance(encoded, tuple):
            bottleneck = encoded[0]
            down_x = encoded[1] if len(encoded) > 1 else []
            if down_x is None:
                down_x = []
            feats = list(down_x)
            if len(feats) == 0 or tuple(feats[-1].shape) != tuple(bottleneck.shape):
                feats.append(bottleneck)
            return feats
        return [encoded]
    if hasattr(model, 'convInit') and hasattr(model, 'down_layers'):
        y = model.convInit(x)
        feats = []
        for down in model.down_layers:
            y = down(y)
            feats.append(y)
        return feats
    raise AttributeError('Could not access SegResNet encoder features.')

def pool_feature_map(feat, mode='gap_gmp'):
    """
    Pools 5D feature map [B,C,D,H,W] to vector.
    """
    feat = feat.float()
    if mode == 'gap':
        return F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
    if mode == 'gmp':
        return F.adaptive_max_pool3d(feat, output_size=1).flatten(1)
    if mode == 'gap_gmp':
        gap = F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
        gmp = F.adaptive_max_pool3d(feat, output_size=1).flatten(1)
        return torch.cat([gap, gmp], dim=1)
    raise ValueError('mode must be gap, gmp, or gap_gmp')

def compute_neighbors(X, max_k):
    """
    Cosine top-k nearest neighbors; self-neighbor excluded.
    """
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argsort(-sim, axis=1)[:, :max_k]
    nn_sim = np.take_along_axis(sim, nn_idx, axis=1)
    return (nn_idx, nn_sim)

@torch.no_grad()
def extract_embedding_from_patch(
    model,
    image_patch_xyz,
    device,
    method='multiscale_gap_gmp',
    *,
    use_amp=False
):
    """
    Extracts one encoder embedding from one ROI patch.

    image_patch_xyz:
    - [4, X, Y, Z]
    """
    image_dhw = xyz_to_dhw(image_patch_xyz)
    x = torch.from_numpy(image_dhw[None]).float().to(device)
    with torch.amp.autocast('cuda', enabled=use_amp):
        feats = encode_segresnet_features(model, x)
        if method == 'bottleneck_gap':
            vectors = [pool_feature_map(feats[-1], mode='gap')]
        elif method == 'bottleneck_gap_gmp':
            vectors = [pool_feature_map(feats[-1], mode='gap_gmp')]
        elif method == 'multiscale_gap':
            vectors = [pool_feature_map(f, mode='gap') for f in feats]
        elif method == 'multiscale_gap_gmp':
            vectors = [pool_feature_map(f, mode='gap_gmp') for f in feats]
        else:
            raise ValueError(f'Unknown embedding method: {method}')
        emb = torch.cat(vectors, dim=1)
    emb = emb.detach().cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def load_encoder(checkpoint_path, device, *, model_kwargs, strict=True):
    """Load trusted local weights and retain checkpoint metadata for audit tables."""
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    model = create_segresnet(model_kwargs)
    result = model.load_state_dict(extract_model_state_dict(checkpoint), strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError("Incomplete checkpoint: missing or unexpected model keys.")
    model.to(device).eval()
    meta = {"checkpoint": str(checkpoint_path),
            "n_params": int(sum(p.numel() for p in model.parameters()))}
    for key in (
        "epoch",
        "round",
        "best_round",
        "best_val_loss",
        "best_val_mean_dice",
        "algorithm",
        "fedprox_mu"
    ):
        if key in checkpoint:
            meta[key] = checkpoint[key]
    cfg = checkpoint.get("cfg", {})
    if isinstance(cfg, dict):
        meta["checkpoint_init_filters"] = cfg.get("init_filters")
        meta["checkpoint_patch_size"] = cfg.get("patch_size")
    return model, meta

class BaselineContext:
    """Patient-aligned arrays and neighbor samplers for notebooks 08b and 08d."""

    def __init__(
        self,
        *,
        patient_ids,
        domain_arr,
        ET_present,
        log1p_WT,
        log1p_TC,
        log1p_ET,
        centroids,
        tumor_feature_z,
        expected_same_by_idx,
        neighbors_df,
        metric_cols
    ):
        self.patient_ids = list(patient_ids)
        self.domain_arr = np.asarray(domain_arr)
        self.ET_present = np.asarray(ET_present)
        self.log1p_WT = np.asarray(log1p_WT)
        self.log1p_TC = np.asarray(log1p_TC)
        self.log1p_ET = np.asarray(log1p_ET)
        self.centroids = centroids
        self.tumor_feature_z = np.asarray(tumor_feature_z)
        self.expected_same_by_idx = np.asarray(expected_same_by_idx)
        self.neighbors_df = neighbors_df
        self.metric_cols = list(metric_cols)
        self.N = len(self.patient_ids)
        if self.N < 2 or len(set(self.patient_ids)) != self.N:
            raise ValueError('At least two unique patients are required.')
        arrays = [
            self.domain_arr,
            self.ET_present,
            self.log1p_WT,
            self.log1p_TC,
            self.log1p_ET,
            self.tumor_feature_z,
            self.expected_same_by_idx,
            *self.centroids.values()
        ]
        if any((len(a) != self.N for a in arrays)):
            raise ValueError('Patient arrays must have identical row counts and order.')
        self.pid_to_idx = {str(pid): i for i, pid in enumerate(self.patient_ids)}
        self.all_indices = np.arange(self.N)
        self.indices_by_domain = {d: np.where(self.domain_arr == d)[0] for d in sorted(set(self.domain_arr))}

    def centroid_distance(self, region, qi, ni):
        a = self.centroids[region][qi]
        b = self.centroids[region][ni]
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            return np.nan
        return float(np.linalg.norm(a - b))

    def pair_metric_dict(self, qi, ni):
        return {
            'same_domain': float(self.domain_arr[qi] == self.domain_arr[ni]),
            'cross_domain': float(self.domain_arr[qi] != self.domain_arr[ni]),
            'same_ET_presence': float(self.ET_present[qi] == self.ET_present[ni]),
            'abs_log1p_WT_diff': float(abs(self.log1p_WT[qi] - self.log1p_WT[ni])),
            'abs_log1p_TC_diff': float(abs(self.log1p_TC[qi] - self.log1p_TC[ni])),
            'abs_log1p_ET_diff': float(abs(self.log1p_ET[qi] - self.log1p_ET[ni])),
            'tumor_feature_z_distance': float(np.linalg.norm(self.tumor_feature_z[qi] - self.tumor_feature_z[ni])),
            'WT_centroid_distance': self.centroid_distance('WT', qi, ni),
            'TC_centroid_distance': self.centroid_distance('TC', qi, ni),
            'ET_centroid_distance': self.centroid_distance('ET', qi, ni)
        }

    def evaluate_sampled_neighbors(self, sampled_neighbors_by_query):
        query_rows = []
        total_pairs = 0
        for qi in range(self.N):
            neigh = np.array(sampled_neighbors_by_query.get(qi, []), dtype=int)
            if len(neigh) == 0:
                continue
            pdf = pd.DataFrame([self.pair_metric_dict(qi, int(ni)) for ni in neigh])
            total_pairs += len(pdf)
            same_domain_rate = float(pdf['same_domain'].mean())
            expected_same = float(self.expected_same_by_idx[qi])
            query_rows.append({
                'query_idx': qi,
                'query_patient_id': self.patient_ids[qi],
                'query_domain': self.domain_arr[qi],
                'effective_k': int(len(neigh)),
                'same_domain_rate': same_domain_rate,
                'cross_domain_rate': float(1.0 - same_domain_rate),
                'expected_same_domain_rate': expected_same,
                'same_domain_enrichment': float(same_domain_rate / expected_same) if expected_same > 0 else np.nan,
                'same_ET_presence_rate': float(pdf['same_ET_presence'].mean()),
                'mean_abs_log1p_WT_diff': float(pdf['abs_log1p_WT_diff'].mean()),
                'mean_abs_log1p_TC_diff': float(pdf['abs_log1p_TC_diff'].mean()),
                'mean_abs_log1p_ET_diff': float(pdf['abs_log1p_ET_diff'].mean()),
                'mean_tumor_feature_z_distance': float(pdf['tumor_feature_z_distance'].mean()),
                'mean_WT_centroid_distance': float(pdf['WT_centroid_distance'].mean(skipna=True)),
                'mean_TC_centroid_distance': float(pdf['TC_centroid_distance'].mean(skipna=True)),
                'mean_ET_centroid_distance': float(pdf['ET_centroid_distance'].mean(skipna=True))
            })
        qdf = pd.DataFrame(query_rows)
        if len(qdf) == 0:
            raise RuntimeError('No sampled neighbors were available.')
        out = {
            'n_queries': int(len(qdf)),
            'n_pairs': int(total_pairs),
            'effective_k_mean': float(qdf['effective_k'].mean()),
            'effective_k_min': int(qdf['effective_k'].min()),
            'effective_k_max': int(qdf['effective_k'].max())
        }
        for c in self.metric_cols:
            out[c] = float(qdf[c].mean(skipna=True)) if c in qdf.columns else np.nan
        return out

    def sample_global_random(self, k, rng):
        sampled = {}
        for qi in range(self.N):
            candidates = np.delete(self.all_indices, qi)
            sampled[qi] = rng.choice(candidates, size=k, replace=False)
        return sampled

    def sample_same_domain_strict(self, k, rng):
        sampled = {}
        for qi in range(self.N):
            d = self.domain_arr[qi]
            candidates = self.indices_by_domain[d]
            candidates = candidates[candidates != qi]
            if len(candidates) == 0:
                sampled[qi] = np.array([], dtype=int)
            else:
                sampled[qi] = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        return sampled

    def get_observed_domain_counts(self, exp_key, retrieval_space, k):
        g = self.neighbors_df[(self.neighbors_df['experiment_key'].astype(str) == str(exp_key)) & (self.neighbors_df['retrieval_space'].astype(str) == str(retrieval_space)) & (self.neighbors_df['neighbor_rank'].astype(int) <= int(k))].copy()
        if len(g) == 0:
            raise ValueError(f'No observed neighbors for {exp_key}, {retrieval_space}, k={k}')
        out = {}
        for qid, qg in g.groupby('query_patient_id'):
            qid = str(qid)
            qi = self.pid_to_idx[qid]
            out[qi] = Counter(qg['neighbor_domain'].astype(str).tolist())
        return out

    def sample_domain_distribution_matched(self, exp_key, retrieval_space, k, rng):
        observed_counts = self.get_observed_domain_counts(exp_key, retrieval_space, k)
        sampled = {}
        for qi in range(self.N):
            counts = observed_counts.get(qi)
            if counts is None:
                raise ValueError(f'Missing observed domain counts for query idx {qi}')
            selected = []
            for d, count in counts.items():
                candidates = self.indices_by_domain[d]
                if d == self.domain_arr[qi]:
                    candidates = candidates[candidates != qi]
                if len(candidates) < count:
                    raise ValueError(f'Cannot sample domain-matched candidates: query={self.patient_ids[qi]}, domain={d}, need={count}, available={len(candidates)}')
                selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
            sampled[qi] = np.array(selected, dtype=int)
        return sampled

    def evaluate_sampled_neighbors_by_query_domain(self, sampled_neighbors_by_query):
        query_rows = []
        for qi in range(self.N):
            neigh = np.array(sampled_neighbors_by_query.get(qi, []), dtype=int)
            if len(neigh) == 0:
                continue
            pdf = pd.DataFrame([self.pair_metric_dict(qi, int(ni)) for ni in neigh])
            same_domain_rate = float(pdf['same_domain'].mean())
            expected_same = float(self.expected_same_by_idx[qi])
            query_rows.append({
                'query_idx': qi,
                'query_patient_id': self.patient_ids[qi],
                'query_domain': self.domain_arr[qi],
                'effective_k': int(len(neigh)),
                'same_domain_rate': same_domain_rate,
                'cross_domain_rate': float(1.0 - same_domain_rate),
                'expected_same_domain_rate': expected_same,
                'same_domain_enrichment': float(same_domain_rate / expected_same) if expected_same > 0 else np.nan,
                'same_ET_presence_rate': float(pdf['same_ET_presence'].mean()),
                'mean_abs_log1p_WT_diff': float(pdf['abs_log1p_WT_diff'].mean()),
                'mean_abs_log1p_TC_diff': float(pdf['abs_log1p_TC_diff'].mean()),
                'mean_abs_log1p_ET_diff': float(pdf['abs_log1p_ET_diff'].mean()),
                'mean_tumor_feature_z_distance': float(pdf['tumor_feature_z_distance'].mean())
            })
        qdf = pd.DataFrame(query_rows)
        return qdf.groupby('query_domain').agg(
            n_queries=('query_patient_id', 'count'),
            effective_k_mean=('effective_k', 'mean'),
            same_domain_rate=('same_domain_rate', 'mean'),
            cross_domain_rate=('cross_domain_rate', 'mean'),
            expected_same_domain_rate=('expected_same_domain_rate', 'mean'),
            same_domain_enrichment=('same_domain_enrichment', 'mean'),
            same_ET_presence_rate=('same_ET_presence_rate', 'mean'),
            mean_abs_log1p_WT_diff=('mean_abs_log1p_WT_diff', 'mean'),
            mean_abs_log1p_TC_diff=('mean_abs_log1p_TC_diff', 'mean'),
            mean_abs_log1p_ET_diff=('mean_abs_log1p_ET_diff', 'mean'),
            mean_tumor_feature_z_distance=('mean_tumor_feature_z_distance', 'mean')
        ).reset_index()
