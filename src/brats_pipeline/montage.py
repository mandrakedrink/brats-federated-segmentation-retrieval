"""PNG montage, lossless master encoding and decoded-pixel verification."""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction
from datetime import datetime
from collections import Counter
import hashlib
import json
import re
import shutil
import subprocess
import uuid
import warnings
import os
import sys
import tempfile
import numpy as np
from PIL import Image as PILImage, ImageDraw
from IPython.display import display
from tqdm.auto import tqdm

CLIP_SPECS = (
    ("Ground truth", "01_ground_truth"),
    ("Centralized S32", "02_centralized_s32"),
    ("FedAvg S32", "03_fedavg_s32"),
    ("FedProx S32", "04_fedprox_s32"),
)
MODES = ("cut", "hold_cut", "fade_black", "crossfade", "wipe")

def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]

def slug(value):
    return re.sub('[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')[:120] or 'case'

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def _ffmpeg_candidates_12(command='auto'):
    candidates = []
    seen = set()

    def add(value):
        if not value:
            return
        value = str(value)
        found = shutil.which(value)
        if not found:
            expanded = Path(value).expanduser()
            if expanded.is_file() and os.access(expanded, os.X_OK):
                found = str(expanded)
        if found:
            found = str(Path(found).resolve())
            if found not in seen:
                seen.add(found)
                candidates.append(found)
    if command and str(command) not in ('auto', 'ffmpeg'):
        add(command)
    for value in ('/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/opt/homebrew/bin/ffmpeg'):
        add(value)
    private_dir = Path.home() / '.cache' / 'brats_montage_12' / 'imageio_ffmpeg_wheel'
    for p in sorted((private_dir / 'imageio_ffmpeg' / 'binaries').glob('ffmpeg*')):
        add(p)
    try:
        import imageio_ffmpeg
        bin_dir = Path(imageio_ffmpeg.__file__).resolve().parent / 'binaries'
        for p in sorted(bin_dir.glob('ffmpeg*')):
            add(p)
        try:
            add(imageio_ffmpeg.get_ffmpeg_exe())
        except (OSError, RuntimeError):
            pass
    except ImportError:
        pass
    add(os.environ.get('IMAGEIO_FFMPEG_EXE'))
    add(command if command and str(command) != 'auto' else 'ffmpeg')
    exe_name = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    for directory in os.get_exec_path():
        if directory:
            add(Path(directory) / exe_name)
    return candidates

def _run_ffmpeg_test_12(command, input_bytes=None):
    result = subprocess.run(
        [str(v) for v in command],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30
    )
    if result.returncode:
        error = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'exit={result.returncode}: {error[-1600:]}')
    return result.stdout

def _test_ffv1_rgba_12(exe):
    width, height, count = (32, 24, 2)
    raw = bytes(((i * 37 + i // 19) % 256 for i in range(width * height * count * 4)))
    prefix = [exe, '-hide_banner', '-loglevel', 'error']
    with tempfile.TemporaryDirectory(prefix='brats_ffmpeg_test_') as tmp:
        video = Path(tmp) / 'rgba_test.mkv'
        _run_ffmpeg_test_12(
            prefix + [
                '-n',
                '-f',
                'rawvideo',
                '-pix_fmt',
                'rgba',
                '-s:v',
                '32x24',
                '-r',
                '12',
                '-i',
                'pipe:0',
                '-an',
                '-frames:v',
                str(count),
                '-filter_threads',
                '1',
                '-threads',
                '1',
                '-c:v',
                'ffv1',
                '-level',
                '3',
                '-coder',
                '1',
                '-g',
                '1',
                '-slicecrc',
                '1',
                '-pix_fmt',
                'bgra',
                str(video)
            ],
            raw
        )
        decoded = _run_ffmpeg_test_12(prefix + [
            '-i',
            str(video),
            '-map',
            '0:v:0',
            '-an',
            '-sn',
            '-dn',
            '-vsync',
            '0',
            '-filter_threads',
            '1',
            '-threads',
            '1',
            '-pix_fmt',
            'rgba',
            '-f',
            'rawvideo',
            'pipe:1'
        ])
    if decoded != raw:
        raise RuntimeError('FFV1 ran, but the RGBA round-trip test failed.')
    return {'status': 'PASS', 'frames': count, 'comparison': 'exact RGBA bytes'}

def find_ffmpeg(command='auto', *, required_encoders=('ffv1', 'libx264')):
    """Ignore missing-library and startup failures while probing FFmpeg."""
    errors = []
    for exe in _ffmpeg_candidates_12(command):
        try:
            version = _run_ffmpeg_test_12([exe, '-version']).decode('utf-8', 'replace')
            if not version.startswith('ffmpeg version'):
                raise RuntimeError('The command did not return an FFmpeg version.')
            listing = _run_ffmpeg_test_12([exe, '-hide_banner', '-encoders']).decode('utf-8', 'replace')
            encoders = set(re.findall('^\\s*[VAS][A-Z.]{5}\\s+(\\S+)', listing, re.M))
            missing = set(required_encoders) - encoders
            if missing:
                raise RuntimeError('FFmpeg runs but lacks encoders: ' + ', '.join(sorted(missing)))
            check = _test_ffv1_rgba_12(exe)
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            message = str(exc)
            errors.append({'path': exe, 'error': message})
            print('SKIP:', exe, '\n ', message.splitlines()[-1][:240])
            continue
        global FFMPEG_INFO_12
        FFMPEG_INFO_12 = {
            'path': exe,
            'version': version.splitlines()[0],
            'encoders': sorted(encoders),
            'ffv1_rgba_test': check,
            'rejected': errors
        }
        print('FFmpeg:', exe)
        print(version.splitlines()[0])
        print('FFV1 RGBA round-trip: PASS (two test frames, including alpha).')
        print(
            'libx264:',
            'YES' if 'libx264' in encoders else 'NO',
            '| libx264rgb:',
            'YES' if 'libx264rgb' in encoders else 'NO'
        )
        return exe
    details = '\n\n'.join((e['path'] + '\n' + e['error'] for e in errors))
    raise RuntimeError('No working FFmpeg was found for PNG montage. The source frames have not been modified.\nRun the optional isolated-wheel installation cell, or set FFMPEG_BIN to a working executable.\n\n' + details)

def _frame_key(path):
    match = re.match('^frame_(\\d+)(?:[_-].*)?\\.png$', path.name, re.I)
    if not match:
        raise ValueError(f'Cannot read frame index: {path.name}')
    return int(match.group(1))

def _angle(path):
    match = re.search('_angle_(-?\\d+(?:\\.\\d+)?)(?:_|\\.png$)', path.name, re.I)
    return float(match.group(1)) if match else None

def list_frames(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'PNG frame directory does not exist:\n{directory}')
    paths = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.name.lower().startswith('frame_') and (p.suffix.lower() == '.png')),
        key=_frame_key
    )
    if not paths:
        raise FileNotFoundError(f'No frame_*.png files in:\n{directory}')
    ids = [_frame_key(p) for p in paths]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Directory {directory} contains multiple PNG files with the same frame index. Separate mixed render runs.')
    expected = list(range(ids[0], ids[-1] + 1))
    if ids != expected:
        missing = sorted(set(expected) - set(ids))
        raise ValueError(f'Missing frame indices in {directory}: {missing[:25]}')
    return paths

def discover_cases(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'SOURCE_ROOT does not exist:\n{root}\nUse the exact output-directory name, including any suffix.')
    cases = []
    for gt in root.rglob('01_ground_truth'):
        if not gt.is_dir():
            continue
        case = gt.parent
        dirs = [case / folder / 'frames' for _, folder in CLIP_SPECS]
        if all((d.is_dir() and any(d.glob('frame_*.png')) for d in dirs)):
            cases.append(case)
    return sorted(set(cases), key=str)

def show_cases(cases):
    if not cases:
        print('No complete clip groups found. Set CASE_DIR or four MANUAL_FRAMES_DIRS.')
    for i, case in enumerate(cases):
        counts = [len(list((case / folder / 'frames').glob('frame_*.png'))) for _, folder in CLIP_SPECS]
        print(f'[{i}] {case}\n    PNG: GT={counts[0]}, Centralized={counts[1]}, FedAvg={counts[2]}, FedProx={counts[3]}')

def choose_folders(cases, *, case_index=None, case_dir=None, manual_frames_dirs=None):
    selected = sum((v is not None for v in (case_index, case_dir, manual_frames_dirs)))
    if selected > 1:
        raise ValueError('Select exactly one of CASE_INDEX, CASE_DIR or MANUAL_FRAMES_DIRS.')
    if manual_frames_dirs is not None:
        if len(manual_frames_dirs) != 4:
            raise ValueError('Exactly four directories are required: GT, Centralized, FedAvg and FedProx.')
        dirs = [Path(p).expanduser().resolve() for p in manual_frames_dirs]
        case_name = dirs[0].parent.parent.name + '_manual'
    else:
        if case_dir is not None:
            case = Path(case_dir).expanduser().resolve()
        elif case_index is not None:
            if not isinstance(case_index, int) or not 0 <= case_index < len(cases):
                raise ValueError('CASE_INDEX is outside the list of available cases.')
            case = cases[case_index]
        elif len(cases) == 1:
            case = cases[0]
        else:
            raise ValueError('Select CASE_INDEX from the list or set CASE_DIR. A patient is not selected arbitrarily.')
        dirs = [case / folder / 'frames' for _, folder in CLIP_SPECS]
        case_name = case.name
    if len(set(dirs)) != 4:
        raise ValueError('Four distinct source directories are required; one directory was repeated.')
    return (dirs, case_name)

def read_png_rgba(path):
    """Only 8-bit truecolor PNGs; reject silent 16->8-bit conversion."""
    path = Path(path)
    with path.open('rb') as handle:
        header = handle.read(29)
    if len(header) < 29 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Unsupported or corrupt PNG: {path}')
    if header[24] != 8 or header[25] not in (2, 6):
        raise ValueError(f'{path.name}: expected 8-bit RGB/RGBA renderer output. Found bit_depth={header[24]}, color_type={header[25]}; automatic bit-depth reduction is disabled.')
    with PILImage.open(path) as image:
        image.load()
        if image.mode not in ('RGB', 'RGBA'):
            raise ValueError(f'Unexpected PNG mode: {image.mode}')
        metadata = {key: image.info[key] for key in (
            'gamma',
            'srgb',
            'chromaticity'
        ) if key in image.info}
        profile = image.info.get('icc_profile')
        if profile:
            metadata['icc_sha256'] = hashlib.sha256(profile).hexdigest()
        rgba = np.array(image.convert('RGBA'), dtype=np.uint8)
    return (rgba, metadata)

def inspect_folders(dirs, *, allow_different_sequences=False):
    clips = []
    for (name, _), directory in zip(CLIP_SPECS, dirs):
        paths = list_frames(directory)
        entries = []
        for path in tqdm(paths, desc=f'Inspecting PNG: {name}'):
            arr, metadata = read_png_rgba(path)
            stat = path.stat()
            entries.append(dict(
                path=str(path),
                index=_frame_key(path),
                angle=_angle(path),
                width=arr.shape[1],
                height=arr.shape[0],
                rgba_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                opaque=bool(np.all(arr[..., 3] == 255)),
                file_size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                color_metadata=metadata
            ))
        sizes = Counter(((e['width'], e['height']) for e in entries))
        print(f'{name}: {len(entries)} frames; sizes {dict(sizes)}')
        print(f'  first: {paths[0].name}\n  last: {paths[-1].name}')
        if len(sizes) > 1:
            warnings.warn(f'{name}: PNG dimensions vary within the clip. Padding only; source tight-crop offsets are not automatically corrected.')
        clips.append(dict(name=name, directory=str(Path(directory).resolve()), frames=entries))
    seqs = [[(e['index'], e['angle']) for e in c['frames']] for c in clips]
    if any((seq != seqs[0] for seq in seqs[1:])):
        message = 'The four clips have different frame indices or angles. Check completeness and camera sequences.'
        if not allow_different_sequences:
            raise ValueError(message + ' To permit this explicitly, set ALLOW_DIFFERENT_SEQUENCES=True.')
        warnings.warn(message)
    profiles = {json.dumps(e['color_metadata'], sort_keys=True) for c in clips for e in c['frames']}
    if len(profiles) > 1:
        warnings.warn('PNG color metadata differ. RGBA values will be preserved, but player color rendering may differ.')
    return clips

def get_canvas(clips, requested=None):
    width = max((e['width'] for c in clips for e in c['frames']))
    height = max((e['height'] for c in clips for e in c['frames']))
    if requested is not None:
        w, h = requested
        if int(w) != w or int(h) != h or w < width or (h < height):
            raise ValueError(f'Canvas must contain every PNG without resizing; minimum {(width, height)}.')
        width, height = (int(w), int(h))
    return (width + width % 2, height + height % 2)

def on_canvas(entry, canvas):
    arr, _ = read_png_rgba(entry['path'])
    if hashlib.sha256(arr.tobytes()).hexdigest() != entry['rgba_sha256']:
        raise RuntimeError(f"PNG changed after inspection: {entry['path']}. Inspect the source files again.")
    W, H = canvas
    h, w = arr.shape[:2]
    if w > W or h > H:
        raise ValueError('PNG does not fit on the canvas; resizing and cropping are not allowed.')
    out = np.zeros((H, W, 4), np.uint8)
    out[..., 3] = 255
    x, y = ((W - w) // 2, (H - h) // 2)
    out[y:y + h, x:x + w] = arr
    return out

def show_endpoints(clips, canvas, out_dir):
    tw, th = (380, 330)
    sheet = PILImage.new('RGB', (2 * tw, len(clips) * (th + 44)), 'black')
    draw = ImageDraw.Draw(sheet)
    for row, clip in enumerate(clips):
        for col, frame in enumerate((clip['frames'][0], clip['frames'][-1])):
            arr = on_canvas(frame, canvas)
            thumb = PILImage.fromarray(arr).convert('RGB')
            thumb.thumbnail((tw, th), PILImage.Resampling.LANCZOS)
            x, y = (col * tw, row * (th + 44))
            draw.text(
                (x + 8, y + 6),
                f"{clip['name']} | {('FIRST' if col == 0 else 'LAST')}",
                fill='white'
            )
            draw.text((x + 8, y + 22), Path(frame['path']).name, fill='white')
            sheet.paste(thumb, (x + (tw - thumb.width) // 2, y + 44 + (th - thumb.height) // 2))
    path = Path(out_dir) / ('endpoints_' + stamp() + '.png')
    sheet.save(path)
    display(sheet)
    return path

def frame_count(seconds, fps):
    seconds = Fraction(str(seconds))
    if seconds < 0:
        raise ValueError('Duration cannot be negative.')
    return int(seconds * fps + Fraction(1, 2))

def build_plan(
    clips,
    mode,
    *,
    fps=12,
    repeat_each=1,
    hold_seconds=0.75,
    transition_seconds=0.35,
    black_seconds=1 / 12,
    start_hold_seconds=0.0,
    final_hold_seconds=1.0
):
    fps = Fraction(str(fps))
    if fps <= 0 or not isinstance(repeat_each, int) or repeat_each < 1:
        raise ValueError('FPS must be positive and REPEAT_EACH_FRAME must be an integer of at least one.')
    if mode not in MODES or not clips or any((not c['frames'] for c in clips)):
        raise ValueError('Unknown mode or empty frame sequence.')
    hold = frame_count(hold_seconds, fps) if mode != 'cut' else 0
    n = frame_count(transition_seconds, fps) if mode in ('fade_black', 'crossfade', 'wipe') else 0
    if mode in ('fade_black', 'crossfade', 'wipe') and n < 1:
        raise ValueError('The transition is shorter than one frame at this FPS.')
    gap = frame_count(black_seconds, fps) if mode == 'fade_black' else 0
    start, end = (frame_count(start_hold_seconds, fps), frame_count(final_hold_seconds, fps))
    operations = []

    def add(kind, count, **kwargs):
        if count:
            operations.append(dict(kind=kind, count=int(count), **kwargs))
    add('still', start, clip=0, frame=0, reason='start_hold')
    for ci, clip in enumerate(clips):
        for fi in range(len(clip['frames'])):
            add('still', repeat_each, clip=ci, frame=fi, reason='source')
        if ci < len(clips) - 1:
            add('still', hold, clip=ci, frame=len(clip['frames']) - 1, reason='join_hold')
            if mode == 'fade_black':
                add('fade_out', n, clip=ci, frame=len(clip['frames']) - 1)
                add('black', gap)
                add('fade_in', n, clip=ci + 1, frame=0)
            elif mode in ('crossfade', 'wipe'):
                add(mode, n, left=ci, right=ci + 1)
    add('still', end, clip=len(clips) - 1, frame=len(clips[-1]['frames']) - 1, reason='final_hold')
    total = sum((o['count'] for o in operations))
    return dict(
        mode=mode,
        fps=str(fps),
        repeat_each=repeat_each,
        hold_frames=hold,
        transition_frames=n,
        black_frames=gap,
        start_hold_frames=start,
        final_hold_frames=end,
        total_frames=total,
        seconds=total / float(fps),
        operations=operations
    )

def _blend(a, b, j, n):
    return ((a.astype(np.uint32) * (n - j) + b.astype(np.uint32) * j + n // 2) // n).astype(np.uint8)

def timeline_frames(clips, canvas, plan):
    W, H = canvas
    black = np.zeros((H, W, 4), np.uint8)
    black[..., 3] = 255
    cached_key, cached = (None, None)

    def load(ci, fi):
        nonlocal cached_key, cached
        if cached_key != (ci, fi):
            cached = on_canvas(clips[ci]['frames'][fi], canvas)
            cached_key = (ci, fi)
        return cached
    for operation in plan['operations']:
        kind, n = (operation['kind'], operation['count'])
        if kind == 'still':
            arr = load(operation['clip'], operation['frame'])
            for _ in range(n):
                yield arr
        elif kind == 'black':
            for _ in range(n):
                yield black
        elif kind in ('fade_out', 'fade_in'):
            arr = load(operation['clip'], operation['frame'])
            for j in range(1, n + 1):
                yield (_blend(arr, black, j, n) if kind == 'fade_out' else _blend(black, arr, j, n + 1))
        elif kind in ('crossfade', 'wipe'):
            left, right = (operation['left'], operation['right'])
            a = load(left, len(clips[left]['frames']) - 1)
            b = load(right, 0)
            for j in range(1, n + 1):
                if kind == 'crossfade':
                    yield _blend(a, b, j, n + 1)
                else:
                    out = a.copy()
                    boundary = W * j // (n + 1)
                    out[:, :boundary] = b[:, :boundary]
                    yield out
        else:
            raise ValueError(kind)

def preview_pair(clips, join, seconds, fps):
    if join not in range(len(clips) - 1):
        raise ValueError('PREVIEW_JOIN must identify an existing clip boundary (0, 1 or 2).')
    n = max(1, frame_count(seconds, Fraction(str(fps))))
    a, b = (clips[join], clips[join + 1])
    return [dict(a, frames=a['frames'][-n:]), dict(b, frames=b['frames'][:n])]

def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)

def verify_frames(video, ffmpeg, canvas, expected_hashes, log_path):
    W, H = canvas
    command = [
        ffmpeg,
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        str(video),
        '-map',
        '0:v:0',
        '-an',
        '-sn',
        '-dn',
        '-vsync',
        '0',
        '-threads',
        '2',
        '-pix_fmt',
        'rgba',
        '-f',
        'rawvideo',
        'pipe:1'
    ]
    with Path(log_path).open('wb') as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            for i, expected in enumerate(tqdm(expected_hashes, desc='Verifying decoded RGBA')):
                raw = _read_exact(proc.stdout, W * H * 4)
                if len(raw) != W * H * 4:
                    raise RuntimeError(f'Decoding ended at frame {i}.')
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise RuntimeError(f'LOSSLESS CHECK FAILED: frame {i}, RGBA values differ.')
            if proc.stdout.read(1):
                raise RuntimeError('The decoder returned extra frames.')
            proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError('Decoding failed. See ' + str(log_path))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout and (not proc.stdout.closed):
                proc.stdout.close()
    return {
        'status': 'PASS',
        'frames': len(expected_hashes),
        'comparison': 'SHA-256 of every decoded RGBA frame'
    }

def _write_all(stream, raw):
    view = memoryview(raw)
    while view:
        n = stream.write(view)
        if not n:
            raise BrokenPipeError('FFmpeg closed the input stream.')
        view = view[n:]

def export_video(
    clips,
    canvas,
    plan,
    ffmpeg,
    out_dir,
    *,
    kind='ffv1',
    verify=True,
    crf=16,
    preset='slow'
):
    if kind not in ('ffv1', 'rgb_lossless_mp4', 'viewing_mp4'):
        raise ValueError(kind)
    encoder = {'ffv1': 'ffv1', 'rgb_lossless_mp4': 'libx264rgb', 'viewing_mp4': 'libx264'}[kind]
    info = globals().get('FFMPEG_INFO_12', {})
    if info.get('path') == str(ffmpeg) and encoder not in info.get('encoders', []):
        raise RuntimeError(f'The selected FFmpeg lacks {encoder}. Install the isolated wheel or select another build. No automatic codec or quality substitution is performed.')
    if kind == 'rgb_lossless_mp4' and (not all((e['opaque'] for c in clips for e in c['frames']))):
        raise ValueError('H.264 RGB cannot preserve alpha. Use FFV1 BGRA for non-opaque PNG files.')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extension = '.mkv' if kind == 'ffv1' else '.mp4'
    path = out_dir / f"{plan['mode']}_{kind}_{stamp()}{extension}"
    partial = path.with_name(path.stem + '.partial' + extension)
    log_path = path.with_suffix('.encode.log')
    W, H = canvas
    command = [
        ffmpeg,
        '-hide_banner',
        '-loglevel',
        'warning',
        '-n',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'rgba',
        '-s:v',
        f'{W}x{H}',
        '-r',
        plan['fps'],
        '-i',
        'pipe:0',
        '-map',
        '0:v:0',
        '-an',
        '-filter_threads',
        '1',
        '-threads',
        '2'
    ]
    if kind == 'ffv1':
        command += [
            '-c:v',
            'ffv1',
            '-level',
            '3',
            '-coder',
            '1',
            '-g',
            '1',
            '-slicecrc',
            '1',
            '-pix_fmt',
            'bgra',
            '-vf',
            'setsar=1'
        ]
    elif kind == 'rgb_lossless_mp4':
        command += [
            '-c:v',
            'libx264rgb',
            '-crf',
            '0',
            '-preset',
            preset,
            '-pix_fmt',
            'rgb24',
            '-vf',
            'setsar=1',
            '-movflags',
            '+faststart'
        ]
    else:
        command += [
            '-c:v',
            'libx264',
            '-crf',
            str(crf),
            '-preset',
            preset,
            '-vf',
            'scale=iw:ih:in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p,setsar=1',
            '-colorspace',
            'bt709',
            '-color_primaries',
            'bt709',
            '-color_trc',
            'iec61966-2-1',
            '-color_range',
            'tv',
            '-movflags',
            '+faststart'
        ]
    command += [str(partial)]
    hashes = []
    write_json(
        path.with_suffix('.plan.json'),
        {
            'canvas': canvas,
            'plan': plan,
            'kind': kind,
            'sources': [c['directory'] for c in clips],
            'command': command
        }
    )
    with log_path.open('wb') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for arr in tqdm(
                timeline_frames(clips, canvas, plan),
                total=plan['total_frames'],
                desc=f'Writing {kind}'
            ):
                if kind == 'viewing_mp4' and (not np.all(arr[..., 3] == 255)):
                    arr = arr.copy()
                    arr[..., :3] = ((arr[..., :3].astype(np.uint32) * arr[
                        ...,
                        3:4
                    ] + 127) // 255).astype(np.uint8)
                    arr[..., 3] = 255
                raw = arr.tobytes()
                if kind != 'viewing_mp4':
                    hashes.append(hashlib.sha256(raw).hexdigest())
                _write_all(proc.stdin, raw)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('FFmpeg exited with an error.\n' + log_path.read_text(errors='replace')[-4000:])
        except BrokenPipeError as exc:
            raise RuntimeError('FFmpeg error. See log:\n' + str(log_path) + '\n' + log_path.read_text(errors='replace')[-4000:]) from exc
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdin and (not proc.stdin.closed):
                proc.stdin.close()
    if kind != 'viewing_mp4' and len(hashes) != plan['total_frames']:
        raise RuntimeError('The submitted frame count does not match the plan.')
    result = {'status': 'NOT_CHECKED', 'frames': plan['total_frames']}
    if verify and kind != 'viewing_mp4':
        result = verify_frames(partial, ffmpeg, canvas, hashes, path.with_suffix('.verify.log'))
    elif kind == 'viewing_mp4':
        result = {'status': 'LOSSY_VIEWING_COPY', 'crf': crf}
    partial.rename(path)
    write_json(path.with_suffix('.verification.json'), result)
    print(f"{result['status']}: {path}\n{plan['total_frames']} frames; {plan['seconds']:.3f} s; {path.stat().st_size / 2 ** 20:.1f} MiB")
    return path

def export_png_timeline(clips, canvas, plan, out_dir):
    directory = Path(out_dir) / ('timeline_png_' + stamp())
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'timeline.json', {'fps': plan['fps'], 'canvas': canvas, 'plan': plan})
    for i, arr in enumerate(tqdm(
        timeline_frames(clips, canvas, plan),
        total=plan['total_frames'],
        desc='Timeline PNGs'
    )):
        PILImage.fromarray(arr).save(directory / f'frame_{i:06d}.png', compress_level=4)
    print('PNG timeline:', directory)
    return directory

def join_three_case_videos(root, kind, indices, *, ffmpeg):
    root = Path(root).expanduser().resolve()
    formats = {
        'viewing_mp4': ('exports', '.mp4'),
        'ffv1': ('masters', '.mkv'),
        'rgb_lossless_mp4': ('masters', '.mp4')
    }
    if kind not in formats:
        raise ValueError(f'Unknown JOIN_KIND: {kind}')
    folder, extension = formats[kind]
    videos = sorted((p for p in root.glob(f'*/*/{folder}/*_{kind}_*{extension}') if p.is_file() and '.partial.' not in p.name))
    if not videos:
        raise FileNotFoundError(f'No completed {kind} in:\n{root}\nCheck JOIN_ROOT and export each case montage first.')
    print('AVAILABLE CASE VIDEOS:')
    for i, p in enumerate(videos):
        print(f'[{i}] {p.parents[2].name}\n    {p.parents[1].name}/{p.name}')
    if indices is None:
        print('\nSet JOIN_INDICES to three indices in the desired order, then run this cell again.')
        return None
    if len(indices) != 3 or len(set(indices)) != 3 or any((not isinstance(
        i,
        int
    ) or not 0 <= i < len(videos) for i in indices)):
        raise ValueError('Select exactly three distinct valid indices.')
    selected = [videos[i] for i in indices]
    case_names = [p.parents[2].name for p in selected]
    case_ids = [re.search('BraTS\\d+_\\d+', name) for name in case_names]
    case_ids = [match.group(0) if match else name for match, name in zip(case_ids, case_names)]
    if len(set(case_ids)) != 3:
        raise ValueError('Multiple versions of the same case were selected. Select one video for each of three distinct cases.')
    metadata = []
    for p in selected:
        plan_file = p.with_suffix('.plan.json')
        if not plan_file.is_file():
            raise FileNotFoundError(f'Missing notebook 12 metadata:\n{plan_file}')
        metadata.append(json.loads(plan_file.read_text(encoding='utf-8')))
    signatures = [(tuple(m['canvas']), Fraction(str(m['plan']['fps'])), m['kind']) for m in metadata]
    if len(set(signatures)) != 1 or signatures[0][2] != kind:
        raise ValueError(f'Video dimensions, FPS or formats differ. Stream-copy concatenation stopped.\nParameters: {signatures}')
    fps = signatures[0][1]
    counts = [int(m['plan']['total_frames']) for m in metadata]
    if fps <= 0 or min(counts) <= 0:
        raise ValueError('Invalid FPS or frame count in .plan.json.')
    if not ffmpeg:
        raise RuntimeError('Pass the verified FFmpeg executable as ffmpeg.')

    def run(args):
        result = subprocess.run([str(ffmpeg), *map(str, args)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-5000:])
        return result.stdout

    def fingerprint(path):
        text = run([
            '-v',
            'error',
            '-xerror',
            '-threads',
            '2',
            '-i',
            path,
            '-map',
            '0:v:0',
            '-an',
            '-sn',
            '-dn',
            '-filter_threads',
            '1',
            '-fps_mode',
            'passthrough',
            '-c:v',
            'rawvideo',
            '-pix_fmt',
            'rgba',
            '-threads',
            '1',
            '-f',
            'framehash',
            '-hash',
            'sha256',
            '-'
        ])
        match = re.search('#tb\\s+0:\\s*(\\S+)', text)
        if not match:
            raise RuntimeError(f'Missing frame time base: {path}')
        time_base = Fraction(match.group(1))
        rows = [line.split(',') for line in text.splitlines() if line.strip() and (not line.startswith('#'))]
        if not rows:
            raise RuntimeError(f'Video contains no frames: {path}')
        times = [int(row[2]) * time_base for row in rows]
        if any((t - times[0] != Fraction(i, 1) / fps for i, t in enumerate(times))):
            raise RuntimeError(f'Frame timestamps do not match {fps} FPS:\n{path}')
        return [row[-1].strip() for row in rows]
    expected_hashes = []
    for name, p, count in zip(case_names, selected, counts):
        print(f'\nVerifying: {name} — {count / float(fps):.2f} s')
        hashes = fingerprint(p)
        if len(hashes) != count:
            raise RuntimeError(f'Video frame count differs from .plan.json:\n{p}')
        expected_hashes.extend(hashes)
    run_dir = root / 'all_cases_combined' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    target = run_dir / ('three_cases' + extension)
    partial = run_dir / ('three_cases.partial' + extension)
    manifest = run_dir / 'concat.ffconcat'
    lines = ['ffconcat version 1.0']
    for p, count in zip(selected, counts):
        if '\n' in str(p) or '\r' in str(p):
            raise ValueError('Newlines in paths are not supported.')
        escaped = p.as_posix().replace("'", "'\\''")
        lines += [f"file '{escaped}'", f'duration {count / float(fps):.9f}']
    manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    command = [
        '-v',
        'error',
        '-n',
        '-f',
        'concat',
        '-safe',
        '0',
        '-i',
        manifest,
        '-map',
        '0:v:0',
        '-c:v',
        'copy',
        '-an',
        '-map_chapters',
        '-1'
    ]
    if extension == '.mp4':
        command += ['-movflags', '+faststart']
    print('\nJoining three cases...')
    run(command + [partial])
    print('Verifying frame content, order and timestamps...')
    if fingerprint(partial) != expected_hashes:
        raise RuntimeError(f'Verification failed; the result was not accepted:\n{partial}')
    partial.rename(target)
    report = {
        'sources': [str(p) for p in selected],
        'fps': str(fps),
        'frames': len(expected_hashes),
        'seconds': len(expected_hashes) / float(fps),
        'verification': 'PASS: decoded RGBA hashes, order and frame cadence',
        'output': str(target)
    }
    (run_dir / 'join_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f"\nCOMPLETE: {report['seconds']:.2f} s; {fps} FPS\nEach case is included once. Verification: PASS.")
    print('Saved:', target)
    return target
