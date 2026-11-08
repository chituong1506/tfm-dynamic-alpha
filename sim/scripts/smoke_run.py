"""Run the Phase 3 200-block fixed and dynamic-alpha smoke simulations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from tfmsim.engine import SimulationConfig, simulate
from tfmsim.estimators import last_block_weighted_mean
from tfmsim.mechanisms import DynamicAlpha, FixedAlpha
from tfmsim.metrics import summarize
from tfmsim.workload import WorkloadConfig, generate_workload


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "sim" / "configs" / "smoke.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    workload_config = WorkloadConfig.from_mapping(config_data["workload"])
    simulation_config = SimulationConfig.from_mapping(config_data["simulation"])
    workload = generate_workload(
        workload_config,
        np.random.default_rng(config_data["seed"]),
    )

    fixed = simulate(workload, FixedAlpha(0.5, 1 / 8), simulation_config)
    dynamic_config = config_data["dynamic_alpha"]
    dynamic = simulate(
        workload,
        DynamicAlpha(
            alpha_max=dynamic_config["alpha_max"],
            kappa=dynamic_config["kappa"],
            eta0=dynamic_config["eta0"],
            gamma=dynamic_config["gamma"],
            estimator=last_block_weighted_mean,
        ),
        simulation_config,
    )

    output = root / "results" / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (output / "git_commit.txt").write_text(f"{revision}\n", encoding="utf-8")
    (output / "assumptions.md").write_text(
        "# Smoke run assumptions\n\n"
        "- Unused fees include priority and compute.\n"
        "- Utilization counts declared objects; prices use the exponential update.\n"
        "- Arrivals are pre-generated at the configured reference price and share identical "
        "transactions and execution outcomes across mechanisms.\n"
        "- The workload uses the calibrated mixture priority distribution; seed and other "
        "parameters are recorded in `config.yaml`.\n",
        encoding="utf-8",
    )
    with (output / "prices.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["block", "mechanism", "object_id", "price"])
        for label, result in (("fixed_alpha", fixed), ("dynamic_alpha", dynamic)):
            for block_index, prices in enumerate(result.price_history):
                for object_id, price in prices.items():
                    writer.writerow([block_index, label, object_id, price])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["mechanism", "metric", "value"],
        )
        writer.writeheader()
        for label, result in (("fixed_alpha", fixed), ("dynamic_alpha", dynamic)):
            for metric, value in summarize(result, simulation_config.target_utilization).items():
                writer.writerow({"mechanism": label, "metric": metric, "value": value})

    figure, axes = plt.subplots(1, len(simulation_config.object_ids), figsize=(12, 3.5), sharex=True)
    for axis, object_id in zip(axes, simulation_config.object_ids, strict=True):
        axis.plot(
            range(len(fixed.price_history)),
            [prices[object_id] for prices in fixed.price_history],
            label="FixedAlpha(0.5, 1/8)",
        )
        axis.plot(
            range(len(dynamic.price_history)),
            [prices[object_id] for prices in dynamic.price_history],
            label="DynamicAlpha",
        )
        axis.set_title(f"Object {object_id}")
        axis.set_xlabel("Block")
        axis.set_ylabel("Price")
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output / "price_over_time.png", dpi=150)
    plt.close(figure)
    (output / "NOTES.md").write_text(
        "# Smoke run\n\n"
        "Ran 200 pre-generated workload blocks with `FixedAlpha(0.5, 1/8)` and "
        "`DynamicAlpha`. The price time series, per-block prices, aggregate metrics, "
        "configuration, assumptions, and source revision are saved in this folder.\n",
        encoding="utf-8",
    )
    print(f"Simulated {len(fixed.records)} blocks per mechanism; results: {output}")


if __name__ == "__main__":
    main()
