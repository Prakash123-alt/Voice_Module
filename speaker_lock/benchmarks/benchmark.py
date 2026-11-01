"""Benchmark entry point (same as `speakerlock benchmark`).

    python scripts/make_testset.py            # once
    python benchmarks/benchmark.py --speakers 8 --out benchmarks/results.json
"""
import sys

from speakerlock.cli import main

sys.exit(main(["benchmark", *sys.argv[1:]]))
