"""ONNX export of the d-vector encoder (needs the [export] extra: torch + resemblyzer)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def export_dvector(dest: str | Path, opset: int = 17, check: bool = True) -> Path:
    """Export Resemblyzer's GE2E VoiceEncoder: input `mels` (batch, 160, 40) -> `embeds` (batch, 256), L2-normed.
    With check=True the ONNX output is compared against torch on random input (max abs diff < 1e-4)."""
    import torch  # noqa: PLC0415
    from resemblyzer import VoiceEncoder  # noqa: PLC0415

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    enc = VoiceEncoder(device="cpu", verbose=False).eval()
    dummy = torch.rand(2, 160, 40) * 0.1
    tmp = dest.with_suffix(".tmp.onnx")
    torch.onnx.export(enc, (dummy,), str(tmp), input_names=["mels"], output_names=["embeds"],
                      dynamic_axes={"mels": {0: "batch"}, "embeds": {0: "batch"}}, opset_version=opset,
                      dynamo=False)
    if check:
        import onnxruntime as ort  # noqa: PLC0415
        x = (torch.rand(3, 160, 40) * 0.2).numpy()
        with torch.no_grad():
            ref = enc(torch.from_numpy(x)).numpy()
        got = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"]).run(None, {"mels": x})[0]
        diff = float(np.abs(ref - got).max())
        if diff > 1e-4:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ONNX export mismatch: max abs diff {diff:.2e}")
    tmp.replace(dest)
    return dest
