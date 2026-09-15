"""Project paths, reproducibility and small artifact helpers."""
from pathlib import Path
import hashlib
import json
import os
import random
import uuid
from datetime import datetime
import numpy as np


def find_project_root(start=None):
    """Find the project containing src/brats_pipeline, or use BRATS_PROJECT_ROOT."""
    override = os.environ.get("BRATS_PROJECT_ROOT")
    start = Path(override or start or Path.cwd()).expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src" / "brats_pipeline").is_dir():
            return candidate
    raise FileNotFoundError(
        "Project root not found. Place notebooks/ and src/ under the same root, "
        "or set BRATS_PROJECT_ROOT to that directory."
    )


class ProjectPaths:
    """Resolve canonical inputs and legacy artifacts saved under notebooks/."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def candidates(self, value):
        if value is None:
            return []
        p = Path(value).expanduser()
        if p.is_absolute():
            paths = [p]
            # Relocate old absolute paths only when the original no longer exists.
            if not p.exists():
                for marker in ("data", "models", "docs", "configs", "logs"):
                    if marker in p.parts:
                        suffix = Path(*p.parts[p.parts.index(marker):])
                        paths += [self.root / suffix, self.root / "notebooks" / suffix]
                        break
        else:
            paths = [self.root / p, self.root / "notebooks" / p]
        result = []
        for path in paths:
            path = path.resolve()
            if path not in result:
                result.append(path)
        return result

    def resolve(self, value, required=True, label="path"):
        """Resolve an artifact using project-root paths as canonical.

        Relative paths are resolved against PROJECT_ROOT first. A legacy
        notebooks/ copy is used only when the canonical project-root artifact
        does not exist. This keeps old notebook-local artifacts readable without
        making duplicate copies ambiguous. Absolute paths are honored exactly
        when they exist.
        """
        candidates = self.candidates(value)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if required:
            raise FileNotFoundError(
                f"Missing {label}: {value}\nChecked:\n"
                + "\n".join(str(p) for p in candidates)
            )
        return None

    def first_existing(self, values):
        """Return the first existing path from an ordered list of path specifications."""
        if not values:
            return None
        for value in values:
            for candidate in self.candidates(value):
                if candidate.exists():
                    return candidate
        return None

    def output(self, value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.root / p


def seed_all(seed=42, *, deterministic=False):
    """Seed Python, NumPy and PyTorch; optionally use deterministic cuDNN settings."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_dataframe(df, name, *, table_dir, index=False):
    path = Path(table_dir) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print("saved:", path)
    return path


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)
    return path


def safe_display(title, value, n=20):
    from IPython.display import display, Markdown
    display(Markdown(f"### {title}"))
    display(value.head(n) if hasattr(value, "head") else value)


def array_digest(array):
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def file_digest(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def signature(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def unique_run_name(prefix="run"):
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def guard_output_directory(directory, *, allow_overwrite=False, patterns=("*.pt",)):
    """Protect existing checkpoints before a new training run starts."""
    directory = Path(directory)
    existing = [p for pattern in patterns for p in directory.glob(pattern)]
    if existing and not allow_overwrite:
        raise FileExistsError(
            f"Existing checkpoints in {directory}. Choose a new save_dir or explicitly "
            "set ALLOW_CHECKPOINT_OVERWRITE=True."
        )
    directory.mkdir(parents=True, exist_ok=True)


def load_array_cache(directory, payload, names):
    """Read an array cache only when its provenance and content hashes match."""
    directory = Path(directory)
    meta_path = directory / "cache.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("signature") != signature(payload):
            return None
        arrays = {}
        for name in names:
            array = np.load(directory / f"{name}.npy", allow_pickle=False)
            if array_digest(array) != meta["arrays"][name]:
                return None
            arrays[name] = array
        return arrays
    except (OSError, ValueError, KeyError):
        return None


def save_array_cache(directory, payload, arrays):
    """Publish cache metadata after all arrays have been written successfully."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, array in arrays.items():
        path = directory / f"{name}.npy"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
        digests[name] = array_digest(array)
    save_json(
        {"signature": signature(payload), "payload": payload, "arrays": digests},
        directory / "cache.json"
    )
