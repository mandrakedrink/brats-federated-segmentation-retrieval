"""Patch sampling, augmentation, loss and federated state aggregation."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from .data import load_case, tumor_center_xyz, crop_patch_xyz, xyz_to_dhw

def random_shifted_center(center_xyz, shape_xyz, max_shift=(32, 32, 24)):
    """Jitter a center and clip its coordinates to the volume bounds."""
    cx, cy, cz = center_xyz
    sx, sy, sz = max_shift
    dx = np.random.randint(-sx, sx + 1)
    dy = np.random.randint(-sy, sy + 1)
    dz = np.random.randint(-sz, sz + 1)
    x = int(np.clip(cx + dx, 0, shape_xyz[0] - 1))
    y = int(np.clip(cy + dy, 0, shape_xyz[1] - 1))
    z = int(np.clip(cz + dz, 0, shape_xyz[2] - 1))
    return (x, y, z)

def random_label_center(seg_xyz, label):
    """Sample one voxel of a class; return None if the class is absent."""
    pts = np.argwhere(seg_xyz == label)
    if len(pts) == 0:
        return None
    idx = np.random.randint(0, len(pts))
    center = pts[idx]
    return tuple(map(int, center))

def random_foreground_center(image_xyz):
    """Sample a nonzero FLAIR voxel, or use the volume center."""
    flair = image_xyz[3]
    fg = flair != 0
    pts = np.argwhere(fg)
    if len(pts) == 0:
        return tuple((int(s // 2) for s in flair.shape))
    idx = np.random.randint(0, len(pts))
    center = pts[idx]
    return tuple(map(int, center))

def random_volume_center(shape_xyz):
    """Sample a center uniformly from the volume."""
    return tuple((int(np.random.randint(0, s)) for s in shape_xyz))

def choose_mixed_center(image_xyz, seg_xyz, p_et=0.2, p_tumor=0.5, p_foreground=0.2, p_random=0.1):
    """Select an ET, tumor, foreground or uniformly sampled crop center."""
    total_p = p_et + p_tumor + p_foreground + p_random
    if not np.isclose(total_p, 1.0):
        raise ValueError(f'Sampling probabilities must sum to 1.0, got {total_p}')
    r = np.random.rand()
    shape_xyz = seg_xyz.shape
    if r < p_et:
        et_center = random_label_center(seg_xyz, label=3)
        if et_center is not None:
            center = random_shifted_center(et_center, shape_xyz=shape_xyz, max_shift=(16, 16, 12))
            mode = 'et_jitter'
        else:
            center = tumor_center_xyz(seg_xyz)
            center = random_shifted_center(center, shape_xyz=shape_xyz, max_shift=(32, 32, 24))
            mode = 'tumor_jitter_no_et'
    elif r < p_et + p_tumor:
        center = tumor_center_xyz(seg_xyz)
        center = random_shifted_center(center, shape_xyz=shape_xyz, max_shift=(32, 32, 24))
        mode = 'tumor_jitter'
    elif r < p_et + p_tumor + p_foreground:
        center = random_foreground_center(image_xyz)
        mode = 'foreground'
    else:
        center = random_volume_center(shape_xyz)
        mode = 'random'
    return (center, mode)

def random_flip_xyz(image_patch_xyz, seg_patch_xyz, flip_prob=0.5):
    """Apply paired random flips to image and segmentation patches."""
    for axis in range(3):
        if np.random.rand() < flip_prob:
            image_patch_xyz = np.flip(image_patch_xyz, axis=axis + 1).copy()
            seg_patch_xyz = np.flip(seg_patch_xyz, axis=axis).copy()
    return (image_patch_xyz, seg_patch_xyz)

def random_intensity_scale_shift(
    image_patch_xyz,
    intensity_prob=0.3,
    scale_range=(0.9, 1.1),
    shift_range=(-0.1, 0.1)
):
    """Apply per-channel intensity augmentation, preserving zero background."""
    image_patch_xyz = image_patch_xyz.copy()
    for c in range(image_patch_xyz.shape[0]):
        if np.random.rand() >= intensity_prob:
            continue
        vol = image_patch_xyz[c]
        mask = vol != 0
        if mask.sum() == 0:
            continue
        scale = np.random.uniform(scale_range[0], scale_range[1])
        shift = np.random.uniform(shift_range[0], shift_range[1])
        vol[mask] = vol[mask] * scale + shift
        image_patch_xyz[c] = vol
    return image_patch_xyz

def random_gaussian_noise(image_patch_xyz, noise_prob=0.2, noise_std=0.03):
    """Add Gaussian noise to nonzero voxels when selected."""
    if np.random.rand() >= noise_prob:
        return image_patch_xyz
    image_patch_xyz = image_patch_xyz.copy()
    for c in range(image_patch_xyz.shape[0]):
        vol = image_patch_xyz[c]
        mask = vol != 0
        if mask.sum() == 0:
            continue
        noise = np.random.normal(0.0, noise_std, size=vol.shape).astype(np.float32)
        vol[mask] = vol[mask] + noise[mask]
        image_patch_xyz[c] = vol
    return image_patch_xyz

def apply_train_augmentations(image_patch_xyz, seg_patch_xyz, *, config):
    """Apply flips, intensity scaling and noise in training order."""
    image_patch_xyz, seg_patch_xyz = random_flip_xyz(
        image_patch_xyz,
        seg_patch_xyz,
        flip_prob=config['flip_prob']
    )
    image_patch_xyz = random_intensity_scale_shift(
        image_patch_xyz,
        intensity_prob=config['intensity_prob'],
        scale_range=config['scale_range'],
        shift_range=config['shift_range']
    )
    image_patch_xyz = random_gaussian_noise(
        image_patch_xyz,
        noise_prob=config['noise_prob'],
        noise_std=config['noise_std']
    )
    return (image_patch_xyz.astype(np.float32), seg_patch_xyz.astype(np.uint8))

class BratsManifestDatasetV2(Dataset):
    """Manifest-backed patch dataset with ET-aware sampling and augmentation."""

    def __init__(
        self,
        df,
        patch_size=(128, 128, 128),
        normalize=True,
        remap=True,
        crop_mode='mixed',
        augment=False,
        *,
        config,
        project_root=None
    ):
        self.df = df.reset_index(drop=True).copy()
        self.patch_size = patch_size
        self.normalize = normalize
        self.remap = remap
        self.crop_mode = crop_mode
        self.augment = augment
        self.config = dict(config)
        self.project_root = project_root

    def __len__(self):
        """Return the number of manifest rows."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load, crop and optionally augment one patient patch."""
        row = self.df.iloc[idx]
        case = load_case(row, normalize=self.normalize, remap=self.remap, project_root=self.project_root)
        image_xyz = case['image_xyz']
        seg_xyz = case['seg_xyz']
        if self.crop_mode == 'mixed':
            center, crop_source = choose_mixed_center(
                image_xyz,
                seg_xyz,
                p_et=self.config['p_et'],
                p_tumor=self.config['p_tumor'],
                p_foreground=self.config['p_foreground'],
                p_random=self.config['p_random']
            )
        elif self.crop_mode == 'tumor':
            center = tumor_center_xyz(seg_xyz)
            crop_source = 'tumor'
        elif self.crop_mode == 'center':
            center = tuple((int(s // 2) for s in seg_xyz.shape))
            crop_source = 'center'
        else:
            raise ValueError("crop_mode must be 'mixed', 'tumor', or 'center'")
        image_patch_xyz, seg_patch_xyz = crop_patch_xyz(
            image_xyz,
            seg_xyz,
            center_xyz=center,
            patch_size=self.patch_size
        )
        if self.augment:
            image_patch_xyz, seg_patch_xyz = apply_train_augmentations(
                image_patch_xyz,
                seg_patch_xyz,
                config=self.config
            )
        image_dhw, seg_dhw = xyz_to_dhw(image_patch_xyz, seg_patch_xyz)
        image_t = torch.from_numpy(image_dhw).float()
        seg_t = torch.from_numpy(seg_dhw).long()
        return {
            'image': image_t,
            'mask': seg_t,
            'patient_id': case['patient_id'],
            'client_domain': case['client_domain'],
            'split': case['split'],
            'center_xyz': center,
            'crop_source': crop_source,
            'augmented': self.augment
        }

class WeightedDiceCELoss(torch.nn.Module):
    """Combine MONAI softmax Dice loss with weighted cross-entropy."""

    def __init__(self, ce_weight=None, dice_weight=1.0, ce_loss_weight=1.0, include_background=True):
        from monai.losses import DiceLoss
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=include_background)
        if ce_weight is not None:
            ce_weight = torch.as_tensor(ce_weight, dtype=torch.float32)
            self.register_buffer('ce_weight', ce_weight)
        else:
            self.ce_weight = None
        self.ce = torch.nn.CrossEntropyLoss(weight=self.ce_weight)
        self.dice_weight = float(dice_weight)
        self.ce_loss_weight = float(ce_loss_weight)

    def forward(self, logits, target):
        """
        logits:
            [B, C, D, H, W]

        target:
            [B, D, H, W] or [B, 1, D, H, W]
        """
        if target.ndim == 4:
            target_ch = target.unsqueeze(1)
        elif target.ndim == 5:
            target_ch = target
        else:
            raise ValueError(f'target must have shape [B,D,H,W] or [B,1,D,H,W], got {tuple(target.shape)}')
        target_ch = target_ch.long()
        target_ce = target_ch.squeeze(1).long()
        dice_loss = self.dice(logits, target_ch)
        ce_loss = self.ce(logits, target_ce)
        return self.dice_weight * dice_loss + self.ce_loss_weight * ce_loss

def dice_binary(pred, target, eps=1e-06):
    """Compute pooled binary Dice for two tensors."""
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().float()
    denom = pred.sum().float() + target.sum().float()
    return float((2.0 * inter + eps) / (denom + eps))

def brats_region_dice(pred_cls, target_cls):
    """Compute WT, TC and ET Dice from contiguous integer labels."""
    out = {}
    out['WT'] = dice_binary(pred_cls > 0, target_cls > 0)
    pred_tc = (pred_cls == 1) | (pred_cls == 3)
    targ_tc = (target_cls == 1) | (target_cls == 3)
    out['TC'] = dice_binary(pred_tc, targ_tc)
    out['ET'] = dice_binary(pred_cls == 3, target_cls == 3)
    return out

def get_round_lr(round_idx, total_rounds, base_lr=0.0001, min_lr=1e-06):
    """
    Cosine decay learning rate by federated communication round.

    round_idx:
    - starts from 1.
    """
    if total_rounds <= 1:
        return base_lr
    t = (round_idx - 1) / (total_rounds - 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * t))
    return float(min_lr + (base_lr - min_lr) * cosine)

def get_cpu_state_dict(model):
    """Clone model parameters and buffers onto CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def fedavg_state_dicts(client_states, client_weights):
    """Weight-average floating tensors; retain the first integer buffer."""
    domains = list(client_states.keys())
    first_domain = domains[0]
    avg_state = {}
    for key in client_states[first_domain].keys():
        first_tensor = client_states[first_domain][key]
        if torch.is_floating_point(first_tensor):
            avg_tensor = torch.zeros_like(first_tensor)
            for domain in domains:
                avg_tensor += client_weights[domain] * client_states[domain][key]
            avg_state[key] = avg_tensor
        else:
            avg_state[key] = first_tensor.clone()
    return avg_state

def make_global_reference_state(global_state, device):
    """Clone the global parameters used by the proximal penalty."""
    return {k: v.detach().to(device) for k, v in global_state.items() if torch.is_floating_point(v)}

def fedprox_l2_penalty(model, global_reference_state):
    """Compute squared L2 distance from the global parameter reference."""
    penalty = None
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ref = global_reference_state.get(name)
        if ref is None:
            continue
        term = torch.sum((param.float() - ref.float()) ** 2)
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return penalty

weighted_average_state_dicts = fedavg_state_dicts

@torch.no_grad()
def validate_patch_level(model, loader, loss_fn, device, max_batches=30, *, use_amp=False):
    """Evaluate tumor-centered patches for checkpoint selection."""
    model.eval()
    losses = []
    dices = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks.unsqueeze(1))
        pred = torch.argmax(logits, dim=1)
        for b in range(pred.shape[0]):
            d = brats_region_dice(pred[b].detach().cpu(), masks[b].detach().cpu())
            dices.append(d)
        losses.append(float(loss.item()))
    mean_dice = {k: float(np.mean([d[k] for d in dices])) if len(dices) > 0 else 0.0 for k in [
        'WT',
        'TC',
        'ET'
    ]}
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else float('nan')
    return (mean_loss, mean_dice)
