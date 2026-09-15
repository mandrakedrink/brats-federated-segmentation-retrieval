from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import re
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion
from skimage.transform import resize
from PIL import Image as PILImage
from tqdm.auto import tqdm
from .common import array_digest, signature, save_json

DEFAULT_REGION_COLORS = {"WT": "#27AE60", "TC": "#9B51E0", "ET": "#F2994A"}
DEFAULT_REGION_NAMES = {"WT": "WT: Whole Tumor", "TC": "TC: Tumor Core", "ET": "ET: Enhancing Tumor"}
RENDER_SCHEMA = 1

class BrainManualCut3D:
    """Voxel MRI cutaways with nested WT, TC and ET display regions."""

    def __init__(
        self,
        img_dim=(140, 140, 90),
        resize_volume=True,
        figsize=(14, 12),
        brain_thr=0.06,
        cmap_name='gray',
        surface_thickness=1,
        crop_bg=False,
        cmap_range=(0.06, 0.78),
        voxel_spacing=(1, 1, 1),
        region_colors=None,
        region_names=None,
        region_order=('WT', 'TC', 'ET'),
        tc_values=(1, 3, 4),
        et_values=(3, 4),
        subplot_adjust=(0.0, 1.0, 0.03, 0.8),
        projection_type='ortho',
        title_fontsize=28,
        legend_fontsize=18,
        title_pad=22,
        legend_bbox=(0.03, 0.98),
        legend_frame_alpha=0.9
    ):
        self.img_dim = tuple(map(int, img_dim))
        self.resize_volume = bool(resize_volume)
        self.figsize = tuple(figsize)
        self.brain_thr = float(brain_thr)
        self.cmap_name = cmap_name
        self.surface_thickness = int(surface_thickness)
        self.crop_bg = bool(crop_bg)
        self.cmap_range = tuple(cmap_range)
        self.voxel_spacing = np.asarray(voxel_spacing, dtype=float)
        self.subplot_adjust = tuple(subplot_adjust)
        self.projection_type = projection_type
        self.region_order = tuple(region_order)
        self.tc_values = tuple(tc_values)
        self.et_values = tuple(et_values)
        self.region_colors = {
            k: to_rgba(v) for k,
            v in (DEFAULT_REGION_COLORS if region_colors is None else region_colors).items()
        }
        self.region_names = dict(DEFAULT_REGION_NAMES if region_names is None else region_names)
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        self.title_pad = title_pad
        self.legend_bbox = tuple(legend_bbox)
        self.legend_frame_alpha = legend_frame_alpha

    @classmethod
    def for_video(cls, **kwargs):
        defaults = dict(resize_volume=False, crop_bg=False)
        defaults.update(kwargs)
        return cls(**defaults)
    for_shorts = for_video

    @staticmethod
    def safe_filename(text, default='brats_mri_3d'):
        text = re.sub('[^A-Za-z0-9_.-]+', '_', str(text).strip()).strip('_')
        return (text or default)[:140]

    def axis_slice(self, value, dim, side='low'):
        if value is None:
            return slice(None)
        if isinstance(value, slice):
            if value.step not in (None, 1):
                raise ValueError('Strided cuts change voxel spacing; use a unit slice step.')
            return value

        def bound(v):
            if v is None:
                return None
            return int(round(v * dim)) if isinstance(v, float) else int(v)
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError('A range cut must contain start and stop.')
            return slice(bound(value[0]), bound(value[1]))
        value = bound(value)
        if side == 'low':
            return slice(0, value)
        if side == 'high':
            return slice(max(dim - value, 0), dim)
        raise ValueError("side must be 'low' or 'high'")

    def seg_to_regions(self, seg):
        return {'WT': seg > 0, 'TC': np.isin(seg, self.tc_values), 'ET': np.isin(seg, self.et_values)}

    def visible_regions(self, seg_cut):
        regions = self.seg_to_regions(seg_cut)
        return [name for name in self.region_order if np.any(regions[name])]

    def make_facecolors(self, brain_cut, seg_cut, surface):
        b = self.norm01(brain_cut)
        low, high = self.cmap_range
        colors = plt.get_cmap(self.cmap_name)(low + b * (high - low))
        colors[..., -1] = 0.0
        colors[surface, -1] = 1.0
        regions = self.seg_to_regions(seg_cut)
        for region in self.region_order:
            colors[regions[region]] = self.region_colors[region]
        return colors

    def add_legend(self, ax, seg_cut=None, loc='upper left', only_visible=True):
        names = self.visible_regions(seg_cut) if only_visible and seg_cut is not None else self.region_order
        handles = [Patch(
            facecolor=self.region_colors[r][:3],
            edgecolor='white',
            label=self.region_names[r]
        ) for r in names]
        if not handles:
            return None
        legend = ax.legend(
            handles=handles,
            loc=loc,
            bbox_to_anchor=self.legend_bbox,
            frameon=True,
            facecolor='black',
            edgecolor='white',
            labelcolor='white',
            fontsize=self.legend_fontsize,
            handlelength=1.8,
            handleheight=1.2,
            borderpad=0.8,
            labelspacing=0.65,
            borderaxespad=0.45
        )
        legend.get_frame().set_alpha(self.legend_frame_alpha)
        return legend

    def set_main_title(self, ax, title, cut, sides, center, show_meta=False):
        ax.set_title(
            self.build_title(title, cut, sides, center, show_meta),
            fontsize=self.title_fontsize,
            fontweight='bold',
            color='white',
            pad=self.title_pad
        )

    def norm01(self, x, p_low=1, p_high=99):
        x = x.astype(np.float32)
        nz = x[x > 0]
        if len(nz) == 0:
            return np.zeros_like(x, dtype=np.float32)
        lo, hi = np.percentile(nz, [p_low, p_high])
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo + 1e-08)

    def enhance(self, x):
        x = self.norm01(x)
        x = np.clip(x - 0.05, 0, 1)
        x = x ** 0.55
        return self.norm01(x)

    def brain_bbox(self, img, pad=3):
        x = self.norm01(img)
        mask = x > self.brain_thr
        pts = np.argwhere(mask)
        if len(pts) == 0:
            raise ValueError('Brain mask is empty. Try lower brain_thr.')
        mn = pts.min(axis=0)
        mx = pts.max(axis=0) + 1
        mn = np.maximum(mn - pad, 0)
        mx = np.minimum(mx + pad, img.shape)
        return (mn, mx)

    def crop_bbox(self, x, bbox):
        mn, mx = bbox
        return x[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]

    def transform_img(self, img):
        """Optionally crop and resample the image before contrast enhancement."""
        bbox = None
        if self.crop_bg:
            bbox = self.brain_bbox(img, pad=3)
            img = self.crop_bbox(img, bbox)
        if self.resize_volume:
            img_out = resize(
                img,
                self.img_dim,
                preserve_range=True,
                anti_aliasing=True
            ).astype(np.float32)
        else:
            img_out = img.astype(np.float32)
        img_out = self.enhance(img_out)
        return (img_out, bbox)

    def transform_seg(self, seg, bbox=None):
        """Apply the image crop and nearest-neighbor label resampling."""
        if bbox is not None:
            seg = self.crop_bbox(seg, bbox)
        if self.resize_volume:
            seg_out = resize(
                seg,
                self.img_dim,
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)
        else:
            seg_out = seg.astype(np.uint8)
        return seg_out

    def tumor_center(self, seg_small):
        pts = np.argwhere(seg_small > 0)
        if len(pts) == 0:
            return tuple(map(int, np.array(seg_small.shape) // 2))
        center = pts.mean(axis=0).astype(int)
        return tuple(map(int, center))

    def make_slices(self, cut=None, sides=('low', 'low', 'low'), shape=None):
        """Build slices from the requested cut and the actual volume shape."""
        if shape is None:
            shape = self.img_dim
        if cut is None:
            return (slice(None), slice(None), slice(None))
        if len(cut) != 3:
            raise ValueError('cut must have 3 values: (x_cut, y_cut, z_cut)')
        sx = self.axis_slice(cut[0], shape[0], sides[0])
        sy = self.axis_slice(cut[1], shape[1], sides[1])
        sz = self.axis_slice(cut[2], shape[2], sides[2])
        return (sx, sy, sz)

    def apply_cut(self, brain, seg_small, cut=None, sides=('low', 'low', 'low')):
        sl = self.make_slices(cut=cut, sides=sides, shape=brain.shape)
        return (brain[sl], seg_small[sl], sl)

    def make_surface(self, brain_cut):
        brain_mask = brain_cut > self.brain_thr
        eroded = binary_erosion(brain_mask, iterations=self.surface_thickness, border_value=0)
        return brain_mask & ~eroded

    def prepare(self, img, seg, cut=None, sides=('low', 'low', 'low')):
        brain, bbox = self.transform_img(img)
        seg_small = self.transform_seg(seg, bbox=bbox)
        center = self.tumor_center(seg_small)
        brain_cut, seg_cut, sl = self.apply_cut(brain, seg_small, cut=cut, sides=sides)
        if 0 in brain_cut.shape:
            raise ValueError('The requested cut is empty.')
        surface = self.make_surface(brain_cut)
        tumor = seg_cut > 0
        filled = surface | tumor
        fc = self.make_facecolors(brain_cut, seg_cut, surface)
        return (brain_cut, seg_cut, filled, fc, sl, center)

    def get_box_aspect(self, brain_cut):
        aspect = np.array(brain_cut.shape, dtype=float) * self.voxel_spacing
        return tuple(aspect)

    def setup_axes(self, fig, ax, brain_cut):
        left, right, bottom, top = self.subplot_adjust
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        if self.projection_type is not None:
            ax.set_proj_type(self.projection_type)
        ax.set_axis_off()
        ax.set_box_aspect(self.get_box_aspect(brain_cut))

    def build_title(self, title, cut, sides, center, show_meta):
        if not show_meta:
            return title
        return f'{title}\ncut={cut}, sides={sides}, tumor_center={center}'

def _valid_png(path):
    try:
        with PILImage.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def _frame_manifest(frames_dir, settings, *, overwrite=False):
    """Only resume a directory whose inputs and settings match this render."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / "render_manifest.json"
    digest = signature(settings)
    existing = list(frames_dir.glob("frame_*.png"))
    previous = json.loads(path.read_text()) if path.is_file() else None
    matching = previous is not None and previous.get("signature") == digest
    if existing and not matching and not overwrite:
        raise RuntimeError(
            f"Render settings are unknown or changed: {frames_dir}. "
            "Use a new output directory or set force_render=True explicitly."
        )
    if overwrite:
        for frame in existing:
            frame.unlink()
    save_json({"schema": RENDER_SCHEMA, "signature": digest, "settings": settings}, path)


def _save_prepared_frame(viz, prepared, out, *, title, cut, sides, angle, elev,
                         show_legend, show_meta, legend_loc, only_visible_legend,
                         dpi, pad_inches):
    brain_cut, seg_cut, filled, facecolors, slices, center = prepared
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=viz.figsize)
        try:
            fig.patch.set_facecolor("black")
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("black")
            ax.voxels(filled, facecolors=facecolors, shade=False,
                      edgecolor="none", linewidth=0)
            ax.view_init(elev=elev, azim=angle)
            viz.set_main_title(ax, title, cut, sides, center, show_meta)
            if show_legend:
                viz.add_legend(ax, seg_cut, loc=legend_loc, only_visible=only_visible_legend)
            viz.setup_axes(fig, ax, brain_cut)
            # Rotation previews retain the original tight-crop output path.
            fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches,
                        facecolor="black", edgecolor="none")
        finally:
            plt.close(fig)
    return str(out)


def save_cutaway_frame(viz, img, seg, out, title="", cut=None,
                       sides=("low", "low", "low"), angle=35, elev=25,
                       show_legend=True, show_meta=False, legend_loc="upper left",
                       only_visible_legend=True, dpi=140, pad_inches=0.03):
    if np.shape(img) != np.shape(seg) or np.ndim(img) != 3:
        raise ValueError("MRI and label arrays must be aligned three-dimensional volumes.")
    prepared = viz.prepare(img, seg, cut=cut, sides=sides)
    return _save_prepared_frame(viz, prepared, out, title=title, cut=cut,
        sides=sides, angle=angle, elev=elev, show_legend=show_legend,
        show_meta=show_meta, legend_loc=legend_loc,
        only_visible_legend=only_visible_legend, dpi=dpi, pad_inches=pad_inches)


def frames_to_mp4(frames, out_mp4, fps=12, quality=9):
    """Encode PNGs in the supplied order. Pad to even dimensions; never resize."""
    if isinstance(frames, (str, Path)):
        frames = sorted(Path(frames).glob("frame_*.png"))
    frames = [Path(p) for p in frames]
    if not frames or fps <= 0:
        raise ValueError("At least one frame and a positive FPS are required.")
    sizes = []
    for path in frames:
        with PILImage.open(path) as image:
            sizes.append(image.size)
    width, height = max(w for w, h in sizes), max(h for w, h in sizes)
    width += width % 2
    height += height % 2
    out = Path(out_mp4)
    if out.suffix.lower() != ".mp4":
        raise ValueError("The viewing video must have an .mp4 extension.")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.mp4")
    with imageio.get_writer(str(partial), format="FFMPEG", fps=fps,
                           codec="libx264", quality=quality, macro_block_size=2) as writer:
        for path in tqdm(frames, desc="Encoding MP4"):
            with PILImage.open(path) as source:
                rgba = source.convert("RGBA")
                canvas = PILImage.new("RGBA", (width, height), (0, 0, 0, 255))
                canvas.alpha_composite(rgba, ((width-rgba.width)//2, (height-rgba.height)//2))
                writer.append_data(np.asarray(canvas.convert("RGB")))
    os.replace(partial, out)
    return str(out)


def render_cutaway_video(viz, img, seg, out_dir, video_name, title="", cut=None,
                         sides=("low", "low", "low"), angle_start=0,
                         angle_end=355, angle_step=5, elev=25, fps=12, dpi=140,
                         show_legend=True, show_meta=False, legend_loc="upper left",
                         only_visible_legend=True, force_render=False):
    """Resume valid frames, then encode a viewing MP4 from the complete sequence."""
    if angle_step <= 0 or angle_end < angle_start:
        raise ValueError("Expected a positive angle step and an increasing angle range.")
    angles = list(range(angle_start, angle_end + 1, angle_step))
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    settings = {"schema": RENDER_SCHEMA, "kind": "rotation",
        "image": array_digest(img), "labels": array_digest(seg),
        "visualizer": vars(viz), "cut": repr(cut), "sides": sides,
        "angles": angles, "elev": elev, "dpi": dpi, "title": title,
        "legend_loc": legend_loc, "show_legend": show_legend,
        "show_meta": show_meta, "only_visible_legend": only_visible_legend}
    _frame_manifest(frames_dir, settings, overwrite=force_render)
    prepared = None
    frame_paths = []
    for index, angle in enumerate(tqdm(angles, desc="Rotation frames")):
        path = frames_dir / f"frame_{index:04d}_angle_{angle:03d}.png"
        if not _valid_png(path):
            if prepared is None:
                prepared = viz.prepare(img, seg, cut=cut, sides=sides)
            _save_prepared_frame(viz, prepared, path, title=title, cut=cut,
                sides=sides, angle=angle, elev=elev, show_legend=show_legend,
                show_meta=show_meta, legend_loc=legend_loc,
                only_visible_legend=only_visible_legend, dpi=dpi, pad_inches=0.03)
        frame_paths.append(path)
    return frames_to_mp4(frame_paths, out_dir/video_name, fps=fps)


@dataclass(frozen=True)
class SweepStyle:
    title: str = "BraTS | GT Regions | X-Cut Z-Sweep"
    angle: float = 35
    elev: float = 25
    dpi: int = 140
    axes_rect: tuple = (0.00, 0.16, 1.00, 0.875)
    volume_zoom: float = 1.50
    legend_anchor: tuple = (0.035, 0.028)
    legend_fontsize: float = 12
    canvas_figsize: tuple = (9.0, 9.0)
    title_y: float = 0.975
    title_fontsize: float = 28

def make_legend_handles(viz):
    return [Patch(
        facecolor=viz.region_colors[region][:3],
        edgecolor='white',
        label=viz.region_names[region]
    ) for region in ['WT', 'TC', 'ET']]

def precompute_xcut_volume(viz, img, seg, x_stop, y_stop):
    brain_full, bbox = viz.transform_img(img)
    seg_full = viz.transform_seg(seg, bbox=bbox)
    x_stop = min(int(x_stop), brain_full.shape[0])
    y_stop = min(int(y_stop), brain_full.shape[1])
    brain_base = brain_full[:x_stop, :y_stop, :].copy()
    seg_base = seg_full[:x_stop, :y_stop, :].copy()
    print('brain_full:', brain_full.shape)
    print('brain_base:', brain_base.shape)
    print('seg labels:', np.unique(seg_base))
    return (brain_base, seg_base)

def get_z_values(z_dim, step=2, direction='top_to_bottom'):
    z_dim = int(z_dim)
    step = int(step)
    if z_dim < 1 or step < 1:
        raise ValueError('Z dimension and step must be positive.')
    if direction == 'top_to_bottom':
        values = list(range(z_dim, 0, -step))
        if values[-1] != 1:
            values.append(1)
        return values
    if direction == 'bottom_to_top':
        values = list(range(1, z_dim + 1, step))
        if values[-1] != z_dim:
            values.append(z_dim)
        return values
    raise ValueError("direction must be 'top_to_bottom' or 'bottom_to_top'")

def make_z_frame(viz, brain_base, seg_base, z):
    z = int(np.clip(z, 1, brain_base.shape[2]))
    brain = brain_base.copy()
    seg = seg_base.copy()
    brain[:, :, z:] = 0
    seg[:, :, z:] = 0
    surface = viz.make_surface(brain)
    tumor = seg > 0
    filled = surface | tumor
    facecolors = viz.make_facecolors(brain, seg, surface)
    return (filled, facecolors)

def save_z_frame(
    viz,
    brain_base,
    seg_base,
    z,
    out_path,
    *,
    title=None,
    angle=None,
    elev=None,
    dpi=None,
    axes_rect=None,
    volume_zoom=None,
    legend_anchor=None,
    legend_fontsize=None,
    style=None
):
    style = SweepStyle() if style is None else style
    title = style.title if title is None else title
    angle = style.angle if angle is None else angle
    elev = style.elev if elev is None else elev
    dpi = style.dpi if dpi is None else dpi
    axes_rect = style.axes_rect if axes_rect is None else axes_rect
    volume_zoom = style.volume_zoom if volume_zoom is None else volume_zoom
    legend_anchor = style.legend_anchor if legend_anchor is None else legend_anchor
    legend_fontsize = style.legend_fontsize if legend_fontsize is None else legend_fontsize
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filled, facecolors = make_z_frame(viz, brain_base, seg_base, z)
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=style.canvas_figsize)
        fig.patch.set_facecolor('black')
        fig.text(
            0.5,
            style.title_y,
            title,
            ha='center',
            va='top',
            fontsize=style.title_fontsize,
            fontweight='bold',
            color='white'
        )
        ax = fig.add_axes(axes_rect, projection='3d')
        ax.set_facecolor('black')
        ax.voxels(filled, facecolors=facecolors, shade=False, edgecolor='none', linewidth=0)
        ax.view_init(elev=elev, azim=angle)
        if getattr(viz, 'projection_type', None) is not None:
            ax.set_proj_type(viz.projection_type)
        ax.set_xlim(0, brain_base.shape[0])
        ax.set_ylim(0, brain_base.shape[1])
        ax.set_zlim(0, brain_base.shape[2])
        ax.set_axis_off()
        aspect = np.array(brain_base.shape, dtype=float)
        if hasattr(viz, 'voxel_spacing'):
            aspect *= np.asarray(viz.voxel_spacing, dtype=float)
        try:
            ax.set_box_aspect(tuple(aspect), zoom=volume_zoom)
        except TypeError:
            ax.set_box_aspect(tuple(aspect))
        legend = fig.legend(
            handles=make_legend_handles(viz),
            loc='lower left',
            bbox_to_anchor=legend_anchor,
            bbox_transform=fig.transFigure,
            frameon=True,
            facecolor='black',
            edgecolor='white',
            labelcolor='white',
            fontsize=legend_fontsize,
            handlelength=1.35,
            handleheight=0.85,
            borderpad=0.45,
            labelspacing=0.3
        )
        legend.get_frame().set_alpha(0.92)
        fig.savefig(out_path, dpi=dpi, facecolor='black', edgecolor='none')
        plt.close(fig)
    return str(out_path)

def render_z_sweep(viz, img, seg, *, frames_dir, video_dir,
                   patient_id, style, x_stop=150, y_stop=240, z_step=2,
                   direction="top_to_bottom", fps=12, force_render=False,
                   hold_start_frames=0, hold_end_frames=0):
    """Fixed camera and XYZ limits; remove layers by decreasing the Z cut boundary."""
    brain_base, seg_base = precompute_xcut_volume(viz, img, seg, x_stop, y_stop)
    z_values = get_z_values(brain_base.shape[2], z_step, direction)
    settings = {"schema": RENDER_SCHEMA, "kind": "z_sweep", "patient_id": patient_id,
                "image": array_digest(img), "labels": array_digest(seg),
                "visualizer": vars(viz), "style": asdict(style),
                "x_stop": x_stop, "y_stop": y_stop, "z_values": z_values}
    _frame_manifest(frames_dir, settings, overwrite=force_render)
    paths = []
    for index, z in enumerate(tqdm(z_values, desc="Z-sweep frames")):
        path = Path(frames_dir) / f"frame_{index:04d}_z_{z:03d}.png"
        if not _valid_png(path):
            save_z_frame(viz, brain_base, seg_base, z, path, style=style)
        paths.append(path)
    sequence = [paths[0]] * hold_start_frames + paths + [paths[-1]] * hold_end_frames
    video = Path(video_dir) / f"{patient_id}_GT_XCut_ZSweep.mp4"
    frames_to_mp4(sequence, video, fps=fps)
    return video, paths


def order_z_frames(frames_dir, direction="top_to_bottom"):
    paths = list(Path(frames_dir).glob("frame_*_z_*.png"))
    if not paths:
        raise FileNotFoundError(f"No Z-sweep PNG files in {frames_dir}")
    def key(path):
        match = re.search(r"_z_(\d+)", path.name)
        if not match:
            raise ValueError(f"Missing Z boundary in {path.name}")
        return int(match.group(1))
    if direction not in ("top_to_bottom", "bottom_to_top"):
        raise ValueError("Unknown sweep direction.")
    return sorted(paths, key=key, reverse=direction == "top_to_bottom")
