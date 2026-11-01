from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

DATA = Path(__file__).parent / "data"


def load(name: str) -> np.ndarray:
    x, sr = sf.read(DATA / f"{name}.wav", dtype="float32")
    assert sr == 16000
    return x


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    s = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    return float(10 * np.log10(np.dot(s, s) / (np.dot(est - s, est - s) + 1e-12)))


@pytest.fixture(scope="session")
def clips() -> dict[str, np.ndarray]:
    return {n: load(n) for n in ("target_enroll", "target_query", "interferer", "third")}


def _voxlock_available() -> bool:
    try:
        from speakerlock.models import voxlock
        voxlock("int8-full")
        return True
    except Exception:
        return False


requires_voxlock = pytest.mark.skipif(not _voxlock_available(), reason="VoxLock model not downloadable")


@pytest.fixture(scope="session")
def voxlock_extractor():
    from speakerlock.extractor import VoxLockExtractor
    return VoxLockExtractor("int8-full", threads=2)
