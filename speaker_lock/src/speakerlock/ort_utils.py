"""onnxruntime session helper (thread count + optional CUDA)."""
from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def session(path: str | Path, threads: int = 1, device: str = "cpu") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(path), so, providers=providers)
