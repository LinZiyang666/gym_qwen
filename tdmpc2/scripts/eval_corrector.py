#!/usr/bin/env python3
"""Evaluate TD-MPC2 with speculative execution and correctors."""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402
from tdmpc2.launch import launch, wrap_dataparallel


VARIANT_ALIASES = {
    "naive3": "3step_naive",
    "spec_corrector": "3step_corrector",
    "spec6_corrector": "6step_corrector",
}


def resolve_variant(args: argparse.Namespace) -> str:
    if args.variant:
        return args.variant
    return VARIANT_ALIASES.get(args.mode, args.mode)


def build_cfg(args: argparse.Namespace) -> Tuple[Any, str]:
    variant = resolve_variant(args)
    cfg_path = Path(args.config) if args.config else REPO_ROOT / "tdmpc2" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.task = args.task or cfg.get("task")
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.eval_episodes = args.episodes
    cfg.checkpoint = args.tdmpc_checkpoint
    cfg.spec_enabled = variant != "baseline"
    cfg.spec_plan_horizon = args.spec_plan_horizon
    cfg.spec_exec_horizon = max(args.spec_exec_horizon, 6) if variant == "6step_corrector" else args.spec_exec_horizon
    cfg.spec_mismatch_threshold = args.spec_mismatch_threshold
    cfg.use_corrector = variant in {"3step_corrector", "6step_corrector"}
    cfg.corrector_ckpt = args.corrector_checkpoint
    cfg.corrector_type = args.corrector_type
    cfg.speculate = False
    cfg = parse_cfg(cfg)
    return cfg, variant


def run_rollout(agent: TDMPC2, env, episodes: int, max_steps: int) -> Dict[str, List[float]]:
    returns: List[float] = []
    lengths: List[int] = []
    replan_counts: List[int] = []
    corrector_steps: List[int] = []
    total_steps = 0
    start = time.time()
    for ep in range(episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done, ep_steps = False, 0
        ep_return = 0.0
        local_corrector_steps = 0
        while not done and (max_steps <= 0 or ep_steps < max_steps):
            action_out = agent.act(
                torch.as_tensor(obs, device=agent.device, dtype=torch.float32),
                t0=ep_steps == 0,
                eval_mode=True,
                return_info=True,
            )
            if isinstance(action_out, tuple) and len(action_out) == 2:
                action, info = action_out
            else:
                action, info = action_out, {}
            next_obs, reward, done, _ = env.step(action.cpu().numpy())
            obs = next_obs[0] if isinstance(next_obs, tuple) else next_obs
            ep_return += float(reward)
            ep_steps += 1
            total_steps += 1
            if isinstance(info, dict) and info.get("used_corrector"):
                local_corrector_steps += 1
        returns.append(ep_return)
        lengths.append(ep_steps)
        replan_counts.append(int(getattr(agent, "episode_replans", 0)))
        corrector_steps.append(int(getattr(agent, "episode_corrector_steps", local_corrector_steps)))
        print(
            f"Episode {ep+1}: return={ep_return:.2f}, steps={ep_steps}, replans={replan_counts[-1]}, "
            f"corrector_steps={corrector_steps[-1]}"
        )
    elapsed = time.time() - start
    steps_per_sec = total_steps / max(elapsed, 1e-6)
    print(f"[throughput] {steps_per_sec:.1f} env steps/sec across {episodes} episodes")
    return {
        "returns": returns,
        "lengths": lengths,
        "replans": replan_counts,
        "corrector_steps": corrector_steps,
    }


def summarize(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    returns = np.array(metrics["returns"])
    lengths = np.array(metrics["lengths"])
    replans = np.array(metrics["replans"])
    corr_steps = np.array(metrics["corrector_steps"])
    return {
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "std_return": float(returns.std()),
        "p5_return": float(np.percentile(returns, 5.0)),
        "mean_length": float(lengths.mean()),
        "mean_replans": float(replans.mean()),
        "mean_corrector_steps": float(corr_steps.mean()),
    }


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    del world_size  # evaluation runs in a single process

    set_seed(args.seed)
    use_gpu = torch.cuda.is_available() and not args.device.startswith("cpu")
    device = torch.device("cuda" if use_gpu else "cpu")

    cfg, inferred_variant = build_cfg(args)
    if use_gpu:
        cfg.device = str(device)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(args.tdmpc_checkpoint)
    agent.eval()

    if use_gpu and torch.cuda.device_count() > 1:
        agent.model = wrap_dataparallel(agent.model)
        if getattr(agent, "corrector", None) is not None:
            agent.corrector = wrap_dataparallel(agent.corrector)

    metrics = run_rollout(agent, env, episodes=args.episodes, max_steps=args.max_steps)
    summary = summarize(metrics)
    variant = args.variant or inferred_variant
    meta = {
        "task": args.task or cfg.task,
        "variant": variant,
        "corrector_type": args.corrector_type,
        "spec_plan_horizon": cfg.spec_plan_horizon,
        "spec_exec_horizon": cfg.spec_exec_horizon,
        "episodes": args.episodes,
        "seed": cfg.seed,
        "tdmpc_checkpoint": args.tdmpc_checkpoint,
        "corrector_checkpoint": args.corrector_checkpoint,
    }
    print("Summary:", summary)
    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{meta['task']}_{meta['variant']}_{meta['corrector_type']}_seed{meta['seed']}"
    base = os.path.join(args.results_dir, f"{run_id}_{ts}")
    with open(base + "_eval.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "summary": summary,
                "episodes": [
                    {
                        "return": float(r),
                        "length": int(l),
                        "num_replans": int(nr),
                        "num_corrector_steps": int(nc),
                    }
                    for r, l, nr, nc in zip(
                        metrics["returns"], metrics["lengths"], metrics["replans"], metrics["corrector_steps"]
                    )
                ],
            },
            f,
            indent=2,
        )
    with open(base + "_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length", "num_replans", "num_corrector_steps"])
        for i, (r, l, nr, nc) in enumerate(
            zip(metrics["returns"], metrics["lengths"], metrics["replans"], metrics["corrector_steps"])
        ):
            writer.writerow([i, r, l, nr, nc])
    if args.output_metrics_path:
        out_dir = os.path.dirname(args.output_metrics_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_metrics_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "metrics": metrics, "summary": summary}, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--env", dest="task", type=str, help="Task name / env id", required=False)
    parser.add_argument("--tdmpc_checkpoint", type=str, required=True, help="Path to TD-MPC2 checkpoint")
    parser.add_argument("--corrector_checkpoint", type=str, default=None, help="Path to trained corrector")
    parser.add_argument("--corrector_type", type=str, default="two_tower", choices=["two_tower", "temporal"])
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "naive3", "spec_corrector", "spec6_corrector"])
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional label for this eval run; if None, infer from cfg/spec settings.",
    )
    parser.add_argument("--spec_plan_horizon", type=int, default=3)
    parser.add_argument("--spec_exec_horizon", type=int, default=3)
    parser.add_argument("--spec_mismatch_threshold", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_metrics_path", type=str, default=None)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/corrector_eval",
        help="Directory to save evaluation metrics (JSON/CSV).",
    )
    parser.add_argument(
        "--gpus", type=str, default="1", help="GPU selection: 'all', N, or comma-separated list",
    )
    args = parser.parse_args()
    valid_variants = {"baseline", "3step_naive", "3step_corrector", "6step_corrector"}
    if args.variant is not None and args.variant not in valid_variants:
        parser.error(f"--variant must be one of {sorted(valid_variants)}")
    return args


if __name__ == "__main__":
    launch(parse_args(), main_worker, use_ddp=False, allow_dataparallel=True)
