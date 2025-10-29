
"""
Convenience launcher for training the Qwen-style Decision Transformer on Humanoid.
Wraps ``experiment_qwen.py`` with a large-config default, while keeping overrides easy.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kick off a Humanoid Decision Transformer run (Qwen backbone). "
                    "Any extra flags can be appended after --."
    )
    parser.add_argument("--env", default="Humanoid-v5", help="Gymnasium env id or D4RL-style name.")
    parser.add_argument("--dataset", default="medium", help="Dataset tag: simple|medium|expert|medium-replay.")
    parser.add_argument("--dataset_dir", default="./data", help="Offline dataset directory fallback.")
    parser.add_argument("--K", type=int, default=20, help="Context length (sequence length).")
    parser.add_argument("--batch_size", type=int, default=32, help=" SGD batch size.")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Transformer hidden size.")
    parser.add_argument("--n_layer", type=int, default=24, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=16, help="Attention heads.")
    parser.add_argument("--n_kv_head", type=int, default=8, help="Key/Value heads (GQA).")
    parser.add_argument("--mlp_ratio", type=float, default=4, help="Feed-forward expansion ratio.")
    parser.add_argument("--learning_rate", type=str, default="1e-4", help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=str, default="0.1", help="AdamW weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="LR warmup steps.")
    parser.add_argument("--num_eval_episodes", type=int, default=3, help="Evaluation episodes per iteration.")
    parser.add_argument("--max_iters", type=int, default=10, help="Training iterations.")
    parser.add_argument("--num_steps_per_iter", type=int, default=1000, help="Gradient steps each iteration.")
    parser.add_argument("--device", default="cuda", help="Training device (cuda/cpu).")
    parser.add_argument("--scale", type=float, default=1000.0, help="RTG scale factor.")
    parser.add_argument("--mode", default="delayed", help="RTG update mode: delayed or normal.")
    parser.add_argument("--target_returns", default=None,
                        help="Optional comma-separated RTG targets (e.g. '7000,9000').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_to_wandb", action="store_true", help="Forward --log_to_wandb to experiment.")
    parser.add_argument("--dry_run", action="store_true", help="Print command without executing.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Any additional flags to pass through to experiment_qwen.py (prefix with --).")
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_py = Path(__file__).with_name("experiment_qwen.py")
    if not experiment_py.exists():
        raise FileNotFoundError(f"Cannot locate {experiment_py}")

    cmd = [
        sys.executable,
        str(experiment_py),
        "--env", args.env,
        "--dataset", args.dataset,
        "--dataset_dir", args.dataset_dir,
        "--model_type", "qwen_dt",
        "--K", str(args.K),
        "--batch_size", str(args.batch_size),
        "--embed_dim", str(args.embed_dim),
        "--n_layer", str(args.n_layer),
        "--n_head", str(args.n_head),
        "--n_kv_head", str(args.n_kv_head),
        "--mlp_ratio", str(args.mlp_ratio),
        "--learning_rate", args.learning_rate,
        "--weight_decay", args.weight_decay,
        "--warmup_steps", str(args.warmup_steps),
        "--num_eval_episodes", str(args.num_eval_episodes),
        "--max_iters", str(args.max_iters),
        "--num_steps_per_iter", str(args.num_steps_per_iter),
        "--device", args.device,
        "--scale", str(args.scale),
        "--mode", args.mode,
        "--seed", str(args.seed),
    ]

    if args.target_returns:
        cmd.extend(["--target_returns", args.target_returns])
    if args.log_to_wandb:
        cmd.append("--log_to_wandb")
    if args.extra_args:
        cmd.extend(args.extra_args)

    print("Launching:", " ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
