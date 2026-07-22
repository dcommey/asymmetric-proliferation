#!/usr/bin/env python3
"""Run all paper experiments and render publication-ready PDF figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from asymprolif.experiments import run
from asymprolif.plotting import build_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--grid-size", type=int, default=61)
    args = parser.parse_args()
    files = run(args.output, size=args.grid_size)
    build_all(args.output)
    print(f"Wrote {len(files)} data files and 9 vector figures to {args.output}")


if __name__ == "__main__":
    main()
