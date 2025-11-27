#!/usr/bin/env python3
"""Plot evaluation summaries for baseline and corrector variants."""

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt

VARIANT_ORDER = ["baseline", "3step_naive", "3step_corrector", "6step_corrector"]


def load_eval_records(paths: List[str], task_filter: str = None) -> List[Dict]:
    records: List[Dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        if task_filter is not None and meta.get("task") != task_filter:
            continue
        summary = payload.get("summary", {})
        records.append(
            {
                "task": meta.get("task"),
                "variant": meta.get("variant"),
                "corrector_type": meta.get("corrector_type", "baseline"),
                "mean_return": summary.get("mean_return"),
                "p5_return": summary.get("p5_return"),
                "mean_replans": summary.get("mean_replans"),
                "mean_corrector_steps": summary.get("mean_corrector_steps"),
            }
        )
    return records


def plot_mean_returns(records: List[Dict], output: str = None) -> None:
    if not records:
        print("No evaluation records to plot.")
        return

    variants = VARIANT_ORDER
    corrector_types = sorted({rec.get("corrector_type") for rec in records})
    x_positions = []
    heights = []
    width = 0.8 / max(len(corrector_types), 1)

    for v_idx, variant in enumerate(variants):
        variant_records = [rec for rec in records if rec.get("variant") == variant]
        if not variant_records:
            continue
        for c_idx, c_type in enumerate(corrector_types):
            match = next((rec for rec in variant_records if rec.get("corrector_type") == c_type), None)
            if match is None:
                continue
            x_positions.append(v_idx + c_idx * width)
            heights.append(match.get("mean_return"))

    plt.figure(figsize=(10, 6))
    plt.bar(x_positions, heights, width=width)
    plt.xticks(
        [i + (width * (len(corrector_types) - 1) / 2) for i in range(len(variants))],
        variants,
    )
    plt.ylabel("Mean return")
    plt.title("Corrector evaluation: mean returns by variant")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(corrector_types, title="Corrector type")
    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        plt.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def print_table(records: List[Dict]) -> None:
    if not records:
        return
    header = ["variant", "corrector_type", "mean_return", "p5_return", "mean_replans", "mean_corrector_steps"]
    print("\t".join(header))
    for rec in records:
        print("\t".join(str(rec.get(k)) for k in header))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_glob",
        type=str,
        required=True,
        help='Glob pattern to locate "*_eval.json" files (e.g., "results/corrector_eval/*_eval.json").',
    )
    parser.add_argument("--output", type=str, default=None, help="Optional output PNG path.")
    parser.add_argument("--task", type=str, default=None, help="Optional task filter for plotting.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.eval_glob))
    if not files:
        print(f"No eval files matched pattern: {args.eval_glob}")
        return

    records = load_eval_records(files, task_filter=args.task)
    if args.task is None and records:
        target_task = records[0].get("task")
        records = [rec for rec in records if rec.get("task") == target_task]
        print(f"No task filter provided; using first task found: {target_task}")

    print_table(records)
    plot_mean_returns(records, output=args.output)


if __name__ == "__main__":
    main()
