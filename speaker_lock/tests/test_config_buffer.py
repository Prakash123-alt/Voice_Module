import numpy as np
import pytest

from speakerlock.buffer import RingBuffer
from speakerlock.config import Config


def test_defaults_valid_and_frame_size():
    cfg = Config().validate()
    assert cfg.frame_samples == 512 and cfg.sample_rate == 16000


def test_from_dict_nested_and_unknown_keys():
    cfg = Config.from_dict({"frame_ms": 40, "gate": {"open_threshold": 0.8, "close_threshold": 0.7}})
    assert cfg.frame_samples == 640 and cfg.gate.open_threshold == 0.8
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_dict({"gate": {"opn": 1}})


def test_hysteresis_order_enforced():
    with pytest.raises(ValueError, match="close_threshold"):
        Config.from_dict({"gate": {"open_threshold": 0.5, "close_threshold": 0.6}})


def test_yaml_roundtrip(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(Config().to_dict()))
    assert Config.from_yaml(p) == Config()


@pytest.mark.parametrize("chunk", [1, 7, 160, 320, 511, 512, 513, 1000, 4096])
def test_ringbuffer_reblocks_any_chunk_size(chunk):
    x = np.arange(10_000, dtype=np.float32)
    rb, frames = RingBuffer(capacity=600), []
    for i in range(0, len(x), chunk):
        rb.push(x[i:i + chunk])
        frames.extend(rb.frames(512))
    out = np.concatenate(frames)
    assert len(out) == (len(x) // 512) * 512
    np.testing.assert_array_equal(out, x[:len(out)])
    assert len(rb) == len(x) % 512


def test_ringbuffer_pop_too_much():
    rb = RingBuffer(4)
    rb.push(np.ones(3))
    with pytest.raises(ValueError):
        rb.pop(4)
