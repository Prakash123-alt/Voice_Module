import pytest

from voice.chunker import SentenceChunker, clean_for_speech


@pytest.mark.parametrize("raw,spoken", [
    ("The capital is **Paris**.", "The capital is Paris."),
    ("- Stick to a *consistent* schedule", "Stick to a consistent schedule"),
    ("## Tips", "Tips"),
    ("See [the docs](https://x.y/z) or https://a.b/c now", "See the docs or link now"),
    ("Run `pip install x` first", "Run pip install x first"),
    ("```\ncode\n```", ""),
])
def test_clean_for_speech(raw, spoken):
    assert clean_for_speech(raw) == spoken


def run(deltas, **kw):
    c = SentenceChunker(**kw)
    out = [s for d in deltas for s in c.feed(d)]
    return out, list(c.flush())


def test_emits_sentence_as_soon_as_it_closes():
    c = SentenceChunker(min_chars=1)
    assert list(c.feed("Hello there")) == []
    assert list(c.feed(". How")) == ["Hello there."]
    assert c.pending == "How"


def test_token_by_token_stream():
    text = "The sky is blue. Grass is green! Is water wet? Yes."
    out, rest = run(list(text), min_chars=1)
    assert out == ["The sky is blue.", "Grass is green!", "Is water wet?"]
    assert rest == ["Yes."]  # last sentence has no trailing space; released on flush


def test_abbreviations_and_initials_do_not_split():
    out, rest = run(["Dr. Smith met Mr. J. Doe, e.g. at noon. Then left. "], min_chars=1)
    assert out == ["Dr. Smith met Mr. J. Doe, e.g. at noon.", "Then left."]
    assert rest == []


def test_decimal_numbers_do_not_split():
    out, rest = run(["Pi is 3.14 roughly. ", "Ok."], min_chars=1)
    assert out == ["Pi is 3.14 roughly."]
    assert rest == ["Ok."]


def test_min_chars_merges_tiny_sentences():
    out, _ = run(["Hi. Ok. This is a longer sentence. "], min_chars=12)
    assert out == ["Hi. Ok. This is a longer sentence."]


def test_closing_quotes_stay_with_sentence():
    out, _ = run(['He said "stop." Then he went home. '], min_chars=1)
    assert out[0] == 'He said "stop."'


def test_max_chars_forces_cut_at_space():
    long = "word " * 100
    out, rest = run([long], min_chars=1, max_chars=50)
    assert out and all(len(s) <= 50 for s in out)
    assert " ".join(out + rest).split() == long.split()


def test_flush_empty():
    assert list(SentenceChunker().flush()) == []
