#!/usr/bin/env python3
"""
Utility script to roll out a trained Qwen Decision Transformer on Humanoid-v5
with real-time rendering.

Usage example:
    python render_humanoid_qwen.py --checkpoint runs/Humanoid-v5_medium_latest.pt --episodes 2
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from experiment_qwen import (
    _canonicalize_env_and_dataset,
    _load_paths,
    _get_state_action_sizes,
    set_seed,
)
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer.models.decision_transformer_qwen3 import (
    DecisionTransformerQwen3 as QwenDecisionTransformer,
)


def _compute_state_stats(paths):
    states = np.concatenate([p["observations"] for p in paths], axis=0)
    state_mean = states.mean(0).astype(np.float32)
    state_std = (states.std(0) + 1e-6).astype(np.float32)
    return state_mean, state_std


def load_model_and_stats(
    checkpoint_path: Path,
    device: str,
    env_override: Optional[str],
    dataset_override: Optional[str],
    dataset_dir_override: Optional[str],
):
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device))
    variant = ckpt.get("variant", {})

    raw_env = env_override or variant.get("env", "Humanoid-v5")
    dataset = dataset_override or variant.get("dataset", "medium")
    dataset_dir = dataset_dir_override or variant.get("dataset_dir", ".")
    gym_env_id, env_base, dataset = _canonicalize_env_and_dataset(raw_env, dataset)

    import gymnasium as gym

    env = gym.make(gym_env_id)
    max_episode_steps = getattr(env.spec, "max_episode_steps", 1000)
    env.close()

    paths = _load_paths(env_base, dataset, gym_env_id, dataset_dir)
    state_dim, act_dim = _get_state_action_sizes(paths)
    state_mean, state_std = _compute_state_stats(paths)

    context_len = int(variant.get("K", 20))

    model = QwenDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        K=context_len,
        embed_dim=int(variant.get("embed_dim", 768)),
        n_layer=int(variant.get("n_layer", 12)),
        n_head=int(variant.get("n_head", 12)),
        n_kv_head=int(variant.get("n_kv_head", 4)),
        mlp_ratio=float(variant.get("mlp_ratio", 4.0)),
        max_timestep=max_episode_steps,
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rtg_scale = float(variant.get("scale", 1000.0))
    mode = variant.get("mode", "delayed")
    target_returns = variant.get("target_returns", [5000.0])

    return (
        model,
        state_mean,
        state_std,
        gym_env_id,
        rtg_scale,
        mode,
        target_returns,
        max_episode_steps,
        context_len,
    )


def _find_latest_checkpoint():
    runs_dir = Path("runs")
    if not runs_dir.exists():
        raise FileNotFoundError("runs/ directory not found. Please provide --checkpoint explicitly.")
    candidates = sorted(
        runs_dir.glob("*_latest.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No *_latest.pt checkpoints found in runs/.")
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Render Humanoid rollout with a trained Qwen DT model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint saved by experiment_qwen.py. Defaults to the most recent runs/*_latest.pt.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for model execution (e.g., cpu or cuda).")
    parser.add_argument("--env", type=str, default=None, help="Override environment ID (defaults to checkpoint variant).")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset tag (defaults to checkpoint variant).")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Override dataset directory (defaults to checkpoint variant).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollout episodes to render.")
    parser.add_argument(
        "--target_return",
        type=float,
        default=None,
        help="Desired return for conditioning. Defaults to the first value stored in the checkpoint.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Return-to-go update mode (delayed or normal). Defaults to checkpoint variant.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help="Gymnasium render mode passed to gym.make (e.g., human, rgb_array).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(int(args.seed))

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    else:
        ckpt_path = _find_latest_checkpoint()
        print(f"[Render] No checkpoint specified, using latest: {ckpt_path}")

    (
        model,
        state_mean,
        state_std,
        gym_env_id,
        rtg_scale,
        default_mode,
        target_returns,
        max_episode_steps,
        context_len,
    ) = (
        load_model_and_stats(
            checkpoint_path=ckpt_path,
            device=args.device,
            env_override=args.env,
            dataset_override=args.dataset,
            dataset_dir_override=args.dataset_dir,
        )
    )

    target_return = (
        float(args.target_return)
        if args.target_return is not None
        else float(target_returns[0]) if target_returns else 5000.0
    )
    eval_mode = args.mode or default_mode

    print(f"[Render] Using env: {gym_env_id}, target_return: {target_return}, mode: {eval_mode}")
    print(f"[Render] Episodes: {args.episodes}, device: {args.device}, render_mode: {args.render_mode}")

    for ep in range(args.episodes):
        ret, length = evaluate_episode_rtg(
            env_name=gym_env_id,
            model=model,
            state_mean=state_mean,
            state_std=state_std,
            target_return=target_return,
            device=args.device,
            rtg_scale=rtg_scale,
            K=context_len,
            max_ep_len=max_episode_steps,
            render=True,
            mode=eval_mode,
            render_mode=args.render_mode,
        )
        print(f"[Episode {ep+1}/{args.episodes}] return={ret:.2f}, length={length}")


if __name__ == "__main__":
    main()
