"""MRI loading and geometry. Arrays use XYZ order until explicit CDHW conversion."""
from pathlib import Path
import json
import numpy as np
from .common import ProjectPaths

def jp(value):
    """Parse a manifest paths mapping without modifying its entries."""
    return value if isinstance(value, dict) else json.loads(value)


def load_nii(path, dtype=np.float32, *, project_root=None):
    """Load NIfTI voxel data without reorienting its array axes."""
    import nibabel as nib
    if project_root is not None:
        path = ProjectPaths(project_root).resolve(path, label="NIfTI volume")
    return nib.load(str(path)).get_fdata().astype(dtype)

def normalize_nonzero(x, eps=1e-08):
    """Z-score strictly positive MRI voxels; leave background at zero."""
    x = x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    mask = x > 0
    if mask.sum() == 0:
        return out
    vals = x[mask]
    mean = vals.mean()
    std = vals.std()
    out[mask] = (x[mask] - mean) / (std + eps)
    return out

def remap_brats_labels(seg):
    """Convert raw BraTS labels {0, 1, 2, 4} to contiguous labels {0, 1, 2, 3}."""
    seg = seg.astype(np.uint8)
    out = np.zeros_like(seg, dtype=np.uint8)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out

def canonicalize_brats_labels(seg):
    """Accept raw labels {0,1,2,4} or contiguous labels {0,1,2,3}."""
    seg = np.asarray(seg).astype(np.uint8)
    out = remap_brats_labels(seg)
    out[seg == 3] = 3
    return out

def load_case(row, normalize=True, remap=True, *, project_root=None):
    """Load four MRI modalities and their segmentation in XYZ order."""
    p = jp(row['paths'])
    images = []
    for m in ['t1', 't1ce', 't2', 'flair']:
        vol = load_nii(p[m], dtype=np.float32, project_root=project_root)
        if normalize:
            vol = normalize_nonzero(vol)
        images.append(vol)
    image_xyz = np.stack(images, axis=0).astype(np.float32)
    seg_xyz = load_nii(p['seg'], dtype=np.uint8, project_root=project_root)
    if remap:
        seg_xyz = remap_brats_labels(seg_xyz)
    return {
        'image_xyz': image_xyz,
        'seg_xyz': seg_xyz,
        'patient_id': row['patient_id'],
        'client_domain': row['client_domain'],
        'split': row['split']
    }

def tumor_center_xyz(seg):
    """Return the integer tumor centroid, with a volume-center fallback."""
    pts = np.argwhere(seg > 0)
    if len(pts) == 0:
        return tuple((int(s // 2) for s in seg.shape))
    center = pts.mean(axis=0).astype(int)
    return tuple(map(int, center))

def axis_crop_pad(center, size, dim):
    """Return source and destination slices for a zero-padded crop."""
    start = int(center - size // 2)
    end = start + size
    src_start = max(start, 0)
    src_end = min(end, dim)
    dst_start = max(0, -start)
    dst_end = dst_start + (src_end - src_start)
    return (slice(src_start, src_end), slice(dst_start, dst_end))

def crop_patch_xyz(image_xyz, seg_xyz, center_xyz, patch_size=(128, 128, 128)):
    """Extract aligned image and label patches with zero padding."""
    px, py, pz = patch_size
    cx, cy, cz = center_xyz
    _, X, Y, Z = image_xyz.shape
    sx, dx = axis_crop_pad(cx, px, X)
    sy, dy = axis_crop_pad(cy, py, Y)
    sz, dz = axis_crop_pad(cz, pz, Z)
    image_patch = np.zeros((4, px, py, pz), dtype=np.float32)
    seg_patch = np.zeros((px, py, pz), dtype=np.uint8)
    image_patch[:, dx, dy, dz] = image_xyz[:, sx, sy, sz]
    seg_patch[dx, dy, dz] = seg_xyz[sx, sy, sz]
    return (image_patch, seg_patch)

def xyz_to_dhw(image_xyz, seg_xyz=None):
    """
    image [C, X, Y, Z] -> [C, Z, Y, X]
    seg   [X, Y, Z]    -> [Z, Y, X]
    """
    image_dhw = np.transpose(image_xyz, (0, 3, 2, 1)).astype(np.float32)
    if seg_xyz is None:
        return image_dhw
    seg_dhw = np.transpose(seg_xyz, (2, 1, 0)).astype(np.uint8)
    return (image_dhw, seg_dhw)

def xyz_to_dhw_image(image_xyz):
    return xyz_to_dhw(image_xyz)


def xyz_to_dhw_seg(seg_xyz):
    return np.transpose(seg_xyz, (2, 1, 0)).astype(np.uint8)


def dhw_to_xyz(seg_dhw):
    return np.transpose(seg_dhw, (2, 1, 0)).astype(np.uint8)

def axis_crop_pad_retrieval(center, patch_dim, full_dim):
    """
    Source and destination slices for one axis crop+pad.
    """
    start = int(round(center - patch_dim / 2))
    end = start + patch_dim
    src_start = max(start, 0)
    src_end = min(end, full_dim)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    return (slice(src_start, src_end), slice(dst_start, dst_end))

def crop_patch_xyz_retrieval(image_xyz, seg_xyz, center_xyz, patch_size=(128, 128, 128)):
    """
    Crops [4, X, Y, Z] image and [X, Y, Z] seg around center_xyz.
    Pads outside volume with zeros.
    """
    px, py, pz = patch_size
    cx, cy, cz = center_xyz
    _, X, Y, Z = image_xyz.shape
    sx, dx = axis_crop_pad_retrieval(cx, px, X)
    sy, dy = axis_crop_pad_retrieval(cy, py, Y)
    sz, dz = axis_crop_pad_retrieval(cz, pz, Z)
    image_patch = np.zeros((image_xyz.shape[0], px, py, pz), dtype=np.float32)
    seg_patch = np.zeros((px, py, pz), dtype=np.uint8)
    image_patch[:, dx, dy, dz] = image_xyz[:, sx, sy, sz]
    seg_patch[dx, dy, dz] = seg_xyz[sx, sy, sz]
    return (image_patch, seg_patch)

def tumor_center_xyz_retrieval(seg_xyz, region='WT'):
    """
    Computes center of a tumor region in XYZ coordinates.

    WT: labels > 0
    TC: labels 1 or 3
    ET: label 3
    """
    if region == 'WT':
        mask = seg_xyz > 0
    elif region == 'TC':
        mask = (seg_xyz == 1) | (seg_xyz == 3)
    elif region == 'ET':
        mask = seg_xyz == 3
    else:
        raise ValueError('region must be WT, TC, or ET')
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return tuple((int(s // 2) for s in seg_xyz.shape))
    center = pts.mean(axis=0)
    return tuple((int(round(v)) for v in center))

def mask_center_xyz(mask_xyz):
    pts = np.argwhere(mask_xyz)
    if len(pts) == 0:
        return None
    return tuple(np.round(pts.mean(axis=0)).astype(int).tolist())

def image_foreground_center_xyz(image_xyz):
    fg = np.any(image_xyz != 0, axis=0)
    center = mask_center_xyz(fg)
    if center is not None:
        return center
    _, X, Y, Z = image_xyz.shape
    return (X // 2, Y // 2, Z // 2)

def euclidean_center_distance(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

def best_tumor_slice_dhw(seg_dhw):
    """Select the depth slice with the largest tumor area."""
    tumor = seg_dhw > 0
    if tumor.sum() == 0:
        return seg_dhw.shape[0] // 2
    counts = tumor.sum(axis=(1, 2))
    return int(np.argmax(counts))

def validate_label_values(seg, *, allow_raw=True):
    allowed = {0, 1, 2, 3, 4} if allow_raw else {0, 1, 2, 3}
    values = set(np.unique(seg).tolist())
    if not values <= allowed:
        raise ValueError(f"Unexpected label values: {sorted(values - allowed)}")


def select_manifest_row(manifest, patient_id):
    rows = manifest.loc[manifest["patient_id"].astype(str) == str(patient_id)]
    if len(rows) != 1:
        raise ValueError(f"Expected one manifest row for {patient_id}, found {len(rows)}")
    return rows.iloc[0]
