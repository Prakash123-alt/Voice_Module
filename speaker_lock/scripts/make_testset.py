"""Build the evaluation set from LibriSpeech dev-clean.

    curl -LO https://www.openslr.org/resources/12/dev-clean.tar.gz && tar xzf dev-clean.tar.gz
    python scripts/make_testset.py --librispeech ./LibriSpeech --out ~/.speakerlock/testset --speakers 8
"""
import argparse

from speakerlock.testset import build

ap = argparse.ArgumentParser()
ap.add_argument("--librispeech", default="~/.speakerlock/data/LibriSpeech")
ap.add_argument("--out", default="~/.speakerlock/testset")
ap.add_argument("--speakers", type=int, default=8)
a = ap.parse_args()
print("manifest:", build(a.librispeech, a.out, a.speakers))
