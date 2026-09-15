"""SegResNet construction, checkpoint loading, sliding-window inference and Dice."""
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import torch
from .data import canonicalize_brats_labels

MODEL_KWARGS = {
    "spatial_dims": 3,
    "in_channels": 4,
    "out_channels": 4,
    "init_filters": 32,
    "blocks_down": (1, 2, 2, 4),
    "blocks_up": (1, 1, 1),
    "dropout_prob": 0.1,
}


def create_segresnet(model_kwargs=None):
    from monai.networks.nets import SegResNet
    return SegResNet(**(MODEL_KWARGS if model_kwargs is None else model_kwargs))


def extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a model state dictionary.")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Unsupported model state dictionary.")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def load_trusted_checkpoint(path, *, map_location="cpu"):
    """Load a local training checkpoint. Only use files from a trusted source."""
    # Training checkpoints contain configuration and history as well as tensors.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


@torch.no_grad()
def sliding_window_logits(model, image_dhw, device, *, roi_size=(128,128,128),
                          sw_batch_size=1, overlap=0.5, mode="gaussian", use_amp=None):
    """Return NCDHW logits without changing preprocessing or array orientation."""
    from monai.inferers import sliding_window_inference
    device = torch.device(device)
    if use_amp is None:
        use_amp = device.type == "cuda"
    model.eval()
    x = torch.from_numpy(np.asarray(image_dhw)).unsqueeze(0).float().to(device)
    with torch.amp.autocast("cuda", enabled=bool(use_amp and device.type == "cuda")):
        logits = sliding_window_inference(x, roi_size=roi_size,
            sw_batch_size=sw_batch_size, predictor=model, overlap=overlap, mode=mode)
    return logits


@torch.no_grad()
def predict_probs_dhw(model, image_dhw, device, **kwargs):
    logits = sliding_window_logits(model, image_dhw, device, **kwargs)
    return torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_labels_dhw(model, image_dhw, device, **kwargs):
    logits = sliding_window_logits(model, image_dhw, device, **kwargs)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def make_pred_with_et_threshold(probs_dhw, et_thr):
    """Choose the best non-ET class, then assign ET where p(ET) >= threshold."""
    if not 0 <= et_thr <= 1:
        raise ValueError("ET threshold must lie in [0, 1].")
    if probs_dhw.ndim != 4 or probs_dhw.shape[0] != 4:
        raise ValueError("Expected probability array [4, D, H, W].")
    pred = np.argmax(probs_dhw[:3], axis=0).astype(np.uint8)
    pred[probs_dhw[3] >= et_thr] = 3
    return pred

def dice_binary_np(pred, target, eps=1e-06):
    """Compute binary Dice; two empty masks receive a score of one."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2.0 * inter + eps) / (denom + eps))

def brats_region_dice_np(pred_cls, target_cls):
    """Compute WT, TC and ET Dice from contiguous integer labels."""
    wt = dice_binary_np(pred_cls > 0, target_cls > 0)
    pred_tc = (pred_cls == 1) | (pred_cls == 3)
    true_tc = (target_cls == 1) | (target_cls == 3)
    tc = dice_binary_np(pred_tc, true_tc)
    et = dice_binary_np(pred_cls == 3, target_cls == 3)
    return {'dice_WT': wt, 'dice_TC': tc, 'dice_ET': et}

def brats_region_voxels(seg_cls):
    seg_cls = canonicalize_brats_labels(seg_cls)
    return {
        'WT_vox': int((seg_cls > 0).sum()),
        'TC_vox': int(((seg_cls == 1) | (seg_cls == 3)).sum()),
        'ET_vox': int((seg_cls == 3).sum())
    }

def make_metric_row(model_name, pred_xyz, target_xyz, et_threshold=None):
    dice = brats_region_dice_np(pred_xyz, target_xyz)
    gt_vol = brats_region_voxels(target_xyz)
    pred_vol = brats_region_voxels(pred_xyz)
    row = {
        'model': model_name,
        **dice,
        'mean_dice': float(np.mean([dice['dice_WT'], dice['dice_TC'], dice['dice_ET']])),
        'gt_WT_vox': gt_vol['WT_vox'],
        'gt_TC_vox': gt_vol['TC_vox'],
        'gt_ET_vox': gt_vol['ET_vox'],
        'pred_WT_vox': pred_vol['WT_vox'],
        'pred_TC_vox': pred_vol['TC_vox'],
        'pred_ET_vox': pred_vol['ET_vox'],
        'WT_ratio': pred_vol['WT_vox'] / max(gt_vol['WT_vox'], 1),
        'TC_ratio': pred_vol['TC_vox'] / max(gt_vol['TC_vox'], 1),
        'ET_ratio': pred_vol['ET_vox'] / max(gt_vol['ET_vox'], 1)
    }
    if et_threshold is not None:
        row['et_threshold'] = float(et_threshold)
    return row
