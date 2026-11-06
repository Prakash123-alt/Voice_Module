"""Configuration loading: firewall.yaml + environment (.env) for secrets.

All thresholds are stored here. None are hardcoded in guard modules.
Thresholds marked as Optional[float] with default None indicate the guard
is in observe mode until scripts/calibrate.py writes the value.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union, get_type_hints

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Per-guard config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EntryConfig:
    bind: str = "127.0.0.1"
    port: int = 7730
    allowed_origins: list[str] = field(default_factory=list)


@dataclass
class SelfSoundConfig:
    aec_backend: str = "webrtc"      # webrtc | gate | none
    loopback_backend: str = "wasapi" # wasapi | stub
    # AEC energy suppression in dB that triggers the "loopback-suspect" source tag.
    loopback_suspect_db: float = 12.0
    # Near-ultrasound detection on the loopback reference audio.
    # Fraction of FFT energy in the top-band proxy above which the flag is set.
    # At 16 kHz this covers the 6.4–8 kHz proxy range.
    # Set to 1.0 to disable.
    near_ultrasound_threshold: float = 0.30


@dataclass
class SpeakerConfig:
    onnx_model: str = "wespeaker-resnet34"  # wespeaker-resnet34 | ecapa-tdnn
    profile_path: str = "profile/owner.npy"
    # None = not yet calibrated → observe mode
    threshold: Optional[float] = None
    # Stricter threshold applied while assistant is speaking. Must be >= threshold.
    # SET BY calibrate.py — never set this looser than threshold.
    strict_threshold: Optional[float] = None
    min_segment_ms: int = 400
    observe_mode: bool = True
    # Optional path to a JSONL score log written during observe mode.
    # Used by scripts/calibrate.py to compute FAR/FRR curves.
    # Leave null to disable score logging.
    score_log_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Enforce the invariant: strict_threshold >= threshold (never looser)."""
        import logging
        if self.threshold is not None and self.strict_threshold is not None:
            if self.strict_threshold < self.threshold:
                # Auto-correct and log — never silently allow a weaker strict gate.
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "SpeakerConfig: strict_threshold (%.4f) < threshold (%.4f) — "
                    "clamping strict_threshold up to threshold.  "
                    "A barge-in must NEVER use a looser threshold.",
                    self.strict_threshold, self.threshold,
                )
                self.strict_threshold = self.threshold


@dataclass
class LiveVoiceConfig:
    enabled: bool = False
    warn_below: float = 0.5
    # Path to an ONNX anti-spoof model (AASIST-compatible).
    # Null = use StubAntispoofModel (score always 0.5).
    # SET THIS to enable real liveness detection.
    model_path: Optional[str] = None


@dataclass
class WordsConfig:
    flag_empty_transcript: bool = True
    flag_injection_patterns: bool = True
    dual_transcribe_on_high_risk: bool = False
    # Background audio injection detection: fraction of energy outside speech band
    # (300–3400 Hz) above which the "background-audio-suspect" flag is raised.
    # 1.0 = disabled; typical clean speech < 0.30; suspicious > 0.60.
    background_inject_threshold: float = 0.65


@dataclass
class ActionConfig:
    confirm_high_risk: bool = True
    confirm_after_untrusted_content: bool = True
    risk_levels: dict[str, list[str]] = field(default_factory=lambda: {
        "low":    ["open_app", "read_text", "navigate"],
        "medium": ["type_text", "search_web", "copy_to_clipboard"],
        "high":   ["send_message", "send_email", "make_payment",
                   "delete_file", "install_app", "change_settings",
                   "execute_command", "access_credentials"],
    })
    # Path to write the JSONL audit log. Null = no audit file (logs only).
    audit_log_path: Optional[str] = None


@dataclass
class GuardsConfig:
    entry:      EntryConfig      = field(default_factory=EntryConfig)
    self_sound: SelfSoundConfig  = field(default_factory=SelfSoundConfig)
    speaker:    SpeakerConfig    = field(default_factory=SpeakerConfig)
    live_voice: LiveVoiceConfig  = field(default_factory=LiveVoiceConfig)
    words:      WordsConfig      = field(default_factory=WordsConfig)
    action:     ActionConfig     = field(default_factory=ActionConfig)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    input_device: Optional[Union[int, str]] = None
    input_gain: float = 1.0

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


@dataclass
class MetricsConfig:
    ring_buffer_size: int = 500   # max samples per stage (bounded memory)
    target_latency_ms: float = 150.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    json: bool = False


@dataclass
class FirewallConfig:
    guards:  GuardsConfig  = field(default_factory=GuardsConfig)
    audio:   AudioConfig   = field(default_factory=AudioConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Loaded from env, never from YAML — not in repr/logs
    service_token: str = field(default="", repr=False)
    hf_token: str      = field(default="", repr=False)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = "firewall.yaml",
        env_file: str | Path = ".env",
    ) -> "FirewallConfig":
        return _load(cls, path, env_file)

    @classmethod
    def defaults(cls) -> "FirewallConfig":
        """Return a config with all defaults (no file I/O). Useful in tests."""
        return cls()


# ---------------------------------------------------------------------------
# Internal loader
# ---------------------------------------------------------------------------

def _is_optional(hint: Any) -> bool:
    return type(None) in getattr(hint, "__args__", ())


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a raw YAML dict.

    - Missing keys keep their dataclass default.
    - None values only applied to Optional fields.
    - Nested dataclass fields are built recursively.
    """
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        raw = data[f.name]
        hint = hints[f.name]
        if raw is None and not _is_optional(hint):
            continue  # keep default
        # Recursively build nested dataclasses
        origin = getattr(hint, "__origin__", None)
        if isinstance(raw, dict) and origin is None:
            try:
                kwargs[f.name] = _build_dataclass(hint, raw)
                continue
            except (TypeError, AttributeError):
                pass
        kwargs[f.name] = raw
    return cls(**kwargs)


def _load(
    cls: type,
    path: str | Path,
    env_file: str | Path,
) -> "FirewallConfig":
    load_dotenv(env_file, override=False)
    raw: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    cfg = _build_dataclass(cls, raw)
    # Secrets from env only — never from YAML
    cfg.service_token = os.environ.get("VFW_SERVICE_TOKEN", "")
    cfg.hf_token = os.environ.get("HF_TOKEN", "")
    return cfg  # type: ignore[return-value]
