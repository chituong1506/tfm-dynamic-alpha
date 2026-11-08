"""Sample and document the calibrated priority distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tfmsim.workload import WorkloadConfig, sample_priorities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "calibration.md",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    config = WorkloadConfig()
    priorities = sample_priorities(args.samples, np.random.default_rng(args.seed), config)
    median = float(np.median(priorities))
    mean = float(priorities.mean())
    p95 = float(np.quantile(priorities, 0.95))
    print(f"median={median:.6f}, mean={mean:.6f}, p95={p95:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# Priority distribution calibration\n\n"
        f"- Samples: {args.samples:,}\n"
        f"- NumPy Generator seed: {args.seed}\n"
        "- Distribution: 50% point mass at priority 1.2 and 50% lognormal.\n"
        "- High-component parameters are derived from conditional mean 8.8 and p90 16.0.\n"
        "- Targets: median approximately 1.2, mean approximately 5, p95 approximately 16.\n"
        f"- Achieved sample median: {median:.6f}\n"
        f"- Achieved sample mean: {mean:.6f}\n"
        f"- Achieved sample p95: {p95:.6f}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
