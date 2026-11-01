"""Configuration loading: config.yaml + environment (.env) for secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union, get_type_hints

import yaml
from dotenv import load_dotenv


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    input_device: Optional[Union[int, str]] = None
    output_device: Optional[Union[int, str]] = None
    input_gain: float = 1.0
    output_gain: float = 1.0

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


@dataclass
class AECConfig:
    enabled: bool = True
    backend: str = "webrtc"
    filter_length_ms: int = 200
    delay_ms: Union[int, str] = "auto"
    max_delay_ms: int = 400
    auto_estimate_min_s: float = 1.0
    gate_when_speaking_if_unavailable: bool = True


@dataclass
class DenoiseConfig:
    backend: str = "rnnoise"


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_speech_ms: int = 120
    hangover_ms: int = 500
    pad_ms: int = 200
    max_segment_s: float = 20.0


@dataclass
class SpeakerConfig:
    backend: str = "resemblyzer"
    profile_path: str = "profile/voice.npy"
    threshold: float = 0.65
    min_segment_ms: int = 400


@dataclass
class BargeInConfig:
    enabled: bool = True
    min_speech_ms: int = 250
    verify_window_ms: int = 1000
    threshold: Optional[float] = None  # relaxed floor for echo-mixed speech; None = speaker.threshold - 0.15


@dataclass
class STTConfig:
    mode: str = "local"
    openrouter_model: str = "google/gemini-2.5-flash"
    local_model: str = "base.en"
    local_compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1                # 1 = greedy (fastest); 5 = more accurate on names
    initial_prompt: str = ""          # context text that biases whisper's vocabulary/spelling
    hotwords: str = ""                # comma-separated names/terms to favour (places, people, jargon)


@dataclass
class LLMConfig:
    model: str = "xiaomi/mimo-v2.6-flash"
    system_prompt: str = "You are a concise voice assistant."
    max_tokens: int = 400
    temperature: float = 0.7
    history_turns: int = 10
    extra_body: dict = field(default_factory=dict)  # passed through to OpenRouter (provider routing, reasoning, ...)


@dataclass
class TTSConfig:
    mode: str = "local"
    openrouter_model: str = "openai/gpt-audio-mini"
    openrouter_voice: str = "alloy"
    piper_model: str = "models/en_US-lessac-medium.onnx"
    piper_speaker: Optional[int] = None
    min_chunk_chars: int = 12


@dataclass
class OpenRouterConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = 30.0
    connect_timeout_s: float = 10.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    api_key: str = field(default="", repr=False)
    site_url: str = ""
    app_name: str = "VoiceModule"


@dataclass
class IsolationConfig:
    enabled: bool = False                    # SpeakerLock gate in front of VAD/STT (needs its own enrollment)
    user_id: str = "me"
    store_dir: str = "~/.speakerlock/users"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    json: bool = False


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    aec: AECConfig = field(default_factory=AECConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _is_optional(hint) -> bool:
    return hint is not None and type(None) in getattr(hint, "__args__", ())


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Build a flat dataclass from a dict, keeping defaults for missing keys and for keys whose
    value is None unless the field is declared Optional."""
    hints = get_type_hints(cls)
    kwargs = {f.name: data[f.name] for f in fields(cls)
              if f.name in data and (data[f.name] is not None or _is_optional(hints.get(f.name)))}
    return cls(**kwargs)


def load_config(path: str | Path = "config.yaml", env_file: str | Path = ".env") -> Config:
    load_dotenv(env_file, override=False)
    raw: dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    hints = get_type_hints(Config)
    cfg = Config(**{f.name: _build(hints[f.name], raw.get(f.name) or {}) for f in fields(Config)})
    cfg.openrouter.api_key = os.environ.get("OPENROUTER_API_KEY", "")
    cfg.openrouter.site_url = os.environ.get("OPENROUTER_SITE_URL", cfg.openrouter.site_url)
    cfg.openrouter.app_name = os.environ.get("OPENROUTER_APP_NAME", cfg.openrouter.app_name)
    return cfg
