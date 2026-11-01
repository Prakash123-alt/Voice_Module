"""Single configuration object for the whole SDK (dataclasses, loadable from YAML)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

SAMPLE_RATE = 16000  # the whole pipeline is 16 kHz mono float32


@dataclass
class VADConfig:
    backend: str = "silero"          # silero | energy
    threshold: float = 0.5           # speech probability
    model_path: Optional[str] = None  # None = bundled/downloaded Silero ONNX


@dataclass
class ExtractorConfig:
    backend: str = "voxlock"         # voxlock | passthrough
    precision: str = "int8-full"     # int8-full (fastest) | int8 | fp16 | fp32
    threads: int = 2                 # onnxruntime intra-op threads for the extractor
    model_dir: Optional[str] = None  # None = download pinned revision from Hugging Face


@dataclass
class VerifierConfig:
    backend: str = "dvector-onnx"    # dvector-onnx (torch-free) | resemblyzer (torch)
    model_path: Optional[str] = None  # ONNX d-vector encoder; None = default location (see models.py)
    threads: int = 1
    window_s: float = 1.6            # rolling window of verified-target speech used for each score
    min_speech_s: float = 0.6        # don't score until this much speech is in the window
    hop_ms: int = 320                # score at most every hop_ms of speech
    max_age_s: float = 8.0           # speech older than this (wall-clock in stream time) leaves the window
    turn_gap_ms: int = 300           # a pause this long starts a new turn: the window restarts (0 = never)
    turn_min_speech_s: float = 0.4   # first verdict of a new turn after this much speech (fast cut-off)
    turn_hop_ms: int = 160           # re-score this often until the turn has min_speech_s of speech


@dataclass
class GateConfig:
    open_threshold: float = 0.72     # score needed to open the gate
    close_threshold: float = 0.62    # gate stays open while score >= this (hysteresis)
    hold_ms: int = 400               # keep open this long after the last passing score (avoids chopping words)
    fade_ms: int = 10                # cross-fade when the gate opens/closes (no clicks)
    max_failures: int = 3            # N consecutive failing windows -> silence + on_unverified
    start_open: bool = False         # True = pass extracted audio before the first score is available
    closed_set: bool = True          # reject if another enrolled user matches better than the target
    other_user_margin: float = 0.03  # closed-set: another user beating the target by this much closes immediately
    strict_turns: bool = True        # close at every new turn until it verifies (no leak; clips ~0.4 s/turn)
    turn_grace_ms: int = 1000        # new turn may pass this long without a confirming score, then closes
    score_ttl_ms: int = 2500         # while someone is talking, a passing score must repeat at least this often
    grace_gain: float = 1.0          # gain during the unverified start of a turn (0 = silence until verified)
    invert: bool = False             # reverse polarity: mute ONLY the enrolled voice (privacy redaction)


@dataclass
class Config:
    frame_ms: int = 32               # 32 ms = 512 samples = Silero's native window
    device: str = "cpu"              # cpu | cuda (onnxruntime CUDA provider, if installed)
    store_dir: str = "~/.speakerlock/users"
    vad: VADConfig = field(default_factory=VADConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    gate: GateConfig = field(default_factory=GateConfig)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def frame_samples(self) -> int:
        return SAMPLE_RATE * self.frame_ms // 1000

    def validate(self) -> "Config":
        g, v = self.gate, self.verifier
        _check(self.frame_ms in (16, 20, 32, 40, 48, 64, 80), "frame_ms must be one of 16/20/32/40/48/64/80")
        _check(self.device in ("cpu", "cuda"), "device must be cpu or cuda")
        _check(self.vad.backend in ("silero", "energy"), "vad.backend must be silero or energy")
        _check(0 < self.vad.threshold < 1, "vad.threshold must be in (0, 1)")
        _check(self.extractor.backend in ("voxlock", "passthrough"), "extractor.backend must be voxlock or passthrough")
        _check(self.extractor.precision in ("int8", "int8-full", "fp16", "fp32"), "bad extractor.precision")
        _check(v.backend in ("dvector-onnx", "resemblyzer"), "verifier.backend must be dvector-onnx or resemblyzer")
        _check(0 < v.min_speech_s <= v.window_s, "need 0 < verifier.min_speech_s <= verifier.window_s")
        _check(0 < v.turn_min_speech_s <= v.window_s and v.turn_hop_ms > 0, "need 0 < turn_min_speech_s <= window_s")
        _check(v.hop_ms > 0 and v.max_age_s >= v.window_s, "need verifier.hop_ms > 0 and max_age_s >= window_s")
        _check(-1 <= g.close_threshold <= g.open_threshold <= 1, "need -1 <= close_threshold <= open_threshold <= 1")
        _check(g.hold_ms >= 0 and g.fade_ms >= 0 and g.max_failures >= 1, "bad gate timing / max_failures")
        _check(g.turn_grace_ms >= 0 and g.score_ttl_ms > 0 and 0.0 <= g.grace_gain <= 1.0,
               "bad gate turn_grace_ms / score_ttl_ms / grace_gain")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data or {}).validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def _check(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None  # type: ignore[misc]
        kwargs[name] = _build(type(default), value or {}) if is_dataclass(default) else value
    return cls(**kwargs)
