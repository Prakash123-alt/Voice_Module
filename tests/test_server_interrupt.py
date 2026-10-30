"""Two-tier interrupt manager: duck on speech, echo guard, verify, backchannel, cooldowns."""
from server.interrupt import InterruptConfig, InterruptManager

CFG = InterruptConfig()


def mk(**kw):
    im = InterruptManager(CFG, **kw)
    im.playback_started()
    for _ in range(11):                      # burn the 200 ms startup cooldown
        im.update(0.0, 0.001, 0.0)
    return im


def test_user_speech_ducks_then_confirms():
    im = mk()
    outs = [im.update(0.9, 0.2, 0.01) for _ in range(10)]
    assert "duck" in outs                                     # tier-1 within ~200 ms
    assert "verify" in [im.update(0.9, 0.2, 0.01) for _ in range(10)]  # 300 ms window
    assert im.verify_result(True, 300) == "confirmed"


def test_echo_only_never_ducks():
    im = mk()
    assert all(im.update(0.9, 0.05, 0.05) is None for _ in range(50))   # mic ≈ ref = echo


def test_other_voice_restores_on_miss():
    im = mk()
    [im.update(0.9, 0.2, 0.01) for _ in range(10)]
    for _ in range(30):
        im.update(0.9, 0.2, 0.01)
        im.verify_result(False, 800)
    outs = {im.update(0.9, 0.2, 0.01) for _ in range(15)}
    assert "restore" in outs or "verify" in outs              # no confirm -> volume back


def test_backchannel_restores_not_confirms():
    im = mk(has_profile=False)
    [im.update(0.9, 0.2, 0.01) for _ in range(10)]            # 200 ms "uh-huh" -> duck
    assert [im.update(0.0, 0.001, 0.0) for _ in range(1)][0] == "restore"


def test_no_profile_confirms_on_duration():
    im = mk(has_profile=False)
    [im.update(0.9, 0.2, 0.01) for _ in range(10)]            # duck
    outs = [im.update(0.9, 0.2, 0.01) for _ in range(10)]     # reach 300 ms
    assert "verify" in outs                                  # caller auto-confirms (no profile)


def test_startup_cooldown_suppresses_early_speech():
    im = InterruptManager(CFG)
    im.playback_started()
    assert all(im.update(0.9, 0.2, 0.01) is None for _ in range(8))  # 160 ms < cooldown
