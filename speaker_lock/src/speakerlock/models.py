"""Model files: pinned sources, SHA-256 verified, cached under $SPEAKERLOCK_HOME (default ~/.speakerlock/models)."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

# VoxLock (Apache-2.0, https://github.com/arbazsiddiqui/voxlock), pinned to a revision published 2026-10-12.
VOXLOCK_REPO = "arbazsiddiqui/voxlock"
VOXLOCK_REVISION = "a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03"
VOXLOCK_SHA256 = {
    "enroll": "86645cba23f207def0b31a6e74a615d3b7a813b4fdf957c53b3af2905ba2ffd3",
    "enroll_fp16": "5fd3bb3179a753e9acc8e1ffeda04fce8ccf5febcd2917956b2419fc71a62533",
    "chunk": "3c31444926cdf3d548935344e2f5f1c71538ded6c2c7d1bd22e2c2141ea0bc2a",
    "chunk_fp16": "d6e931ea261d217d03bdad05aa8e1d6a11a41d4a421a90ea91622d8ec17f8e82",
    "chunk_int8": "f0d2b0aaf95affe8894ea008d0105be4f5f5354f7244835534c31b1a7dd4699a",
    "chunk_int8_full": "7b57c1bfd5acd5b33768a9dff09d9af487d45374b3ba0b991e541ff121eaa337",
}
VOXLOCK_PRECISIONS = {  # precision -> (enroll file, chunk file)
    "int8": ("enroll", "chunk_int8"),
    "int8-full": ("enroll", "chunk_int8_full"),
    "fp16": ("enroll_fp16", "chunk_fp16"),
    "fp32": ("enroll", "chunk"),
}

DVECTOR_FILE = "dvector_ge2e.onnx"  # produced by scripts/export_onnx.py from Resemblyzer's GE2E encoder


def home() -> Path:
    p = Path(os.environ.get("SPEAKERLOCK_HOME", "~/.speakerlock")).expanduser() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verified(path: Path, expected: str) -> Path:
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"checksum mismatch for {path.name}: expected {expected[:12]}..., got {got[:12]}...")
    return path


def silero_vad() -> Path:
    """Silero VAD ONNX: from the installed `silero_vad` package if present, else downloaded once."""
    try:
        import silero_vad  # noqa: PLC0415
        p = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
        if p.exists() and sha256(p) == SILERO_SHA256:
            return p
    except ImportError:
        pass
    dest = home() / "silero_vad.onnx"
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(SILERO_URL, tmp)  # noqa: S310 - fixed https URL
        _verified(tmp, SILERO_SHA256).replace(dest)
    return _verified(dest, SILERO_SHA256)


def voxlock(precision: str = "int8-full") -> tuple[Path, Path]:
    """(enroll.onnx, chunk.onnx) for `precision`, downloaded from the pinned Hugging Face revision."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    names = VOXLOCK_PRECISIONS[precision]
    paths = [Path(hf_hub_download(VOXLOCK_REPO, f"onnx/{n}.onnx", revision=VOXLOCK_REVISION)) for n in names]
    return tuple(_verified(p, VOXLOCK_SHA256[n]) for p, n in zip(paths, names))  # type: ignore[return-value]


def dvector(path: str | None = None, auto_export: bool = True) -> Path:
    """ONNX GE2E d-vector encoder. Exported on first use when torch + resemblyzer are installed."""
    p = Path(path).expanduser() if path else home() / DVECTOR_FILE
    if p.exists():
        return p
    if not auto_export:
        raise FileNotFoundError(f"{p} missing - run `python scripts/export_onnx.py` (needs the [export] extra)")
    from .export import export_dvector  # noqa: PLC0415
    return export_dvector(p)
