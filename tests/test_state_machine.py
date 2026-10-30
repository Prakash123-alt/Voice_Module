import pytest

from voice.state import InvalidTransition, State, StateMachine, TRANSITIONS


def test_initial_state_is_idle():
    assert StateMachine().state is State.IDLE


def test_happy_path():
    sm = StateMachine()
    for s in (State.LISTENING, State.PROCESSING, State.SPEAKING, State.LISTENING):
        sm.transition(s)
    assert sm.state is State.LISTENING
    assert [h[1] for h in sm.history] == [State.LISTENING, State.PROCESSING, State.SPEAKING, State.LISTENING]


def test_barge_in_path():
    sm = StateMachine()
    sm.transition(State.LISTENING); sm.transition(State.PROCESSING); sm.transition(State.SPEAKING)
    sm.transition(State.INTERRUPTED, "barge-in")
    sm.transition(State.LISTENING)
    assert sm.state is State.LISTENING


@pytest.mark.parametrize("bad", [State.PROCESSING, State.SPEAKING, State.INTERRUPTED])
def test_invalid_from_idle(bad):
    with pytest.raises(InvalidTransition):
        StateMachine().transition(bad)


def test_cannot_interrupt_when_not_speaking():
    sm = StateMachine()
    sm.transition(State.LISTENING)
    assert not sm.can(State.INTERRUPTED)
    assert sm.try_transition(State.INTERRUPTED) is False
    assert sm.state is State.LISTENING


def test_interrupted_must_return_to_listening_or_idle():
    assert TRANSITIONS[State.INTERRUPTED] == {State.LISTENING, State.IDLE}


def test_on_change_callback_and_shutdown_from_any_state():
    seen = []
    sm = StateMachine(on_change=lambda a, b: seen.append((a, b)))
    sm.transition(State.LISTENING); sm.transition(State.IDLE)
    assert seen == [(State.IDLE, State.LISTENING), (State.LISTENING, State.IDLE)]
    for s in State:
        assert State.IDLE in TRANSITIONS[s] or s is State.IDLE
