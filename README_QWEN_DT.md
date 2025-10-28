
# Qwen3-Style Decision Transformer (for MuJoCo Humanoid)

## Quick Start

1. (Optional) Create the Conda environment from `environment.yaml`.

2. Download the Humanoid offline datasets:
   ```bash
   python ./data/download_d4rl_datasets.py --env humanoid --out ./data
   ```
   **Heads-up:** the downloader may produce file names that differ from what `experiment_qwen.py` expects (`humanoid-medium-v2.pkl`, `humanoid-medium.pkl`, etc.). If you see variants such as `humanoid_medium-v2.pkl`, rename them to the snake-case names used in this repo (e.g., `humanoid-medium-v2.pkl`) so the training script can locate them automatically.

3. Train (≈0.6B config; requires a strong GPU):
   ```bash
   python train_humanoid_qwen_dt.py
   ```

   For a smaller sanity check:
   ```bash
   python experiment_qwen.py --env Humanoid-v5 --dataset medium   --target_returns 5000,8000   --embed_dim 768 --n_layer 12 --n_head 12 --n_kv_head 4 --mlp_ratio 4.0   --batch_size 64 --K 20 --max_iters 1 --num_steps_per_iter 200   --scale 1000.0 --mode delayed
   ```

## Rendering & Evaluation
- After training, run `render_humanoid_qwen.py` to load the latest `runs/*_latest.pt` checkpoint and watch a rollout:
  ```bash
  python render_humanoid_qwen.py --episodes 1 --render_mode human --device cpu
  ```
  The script auto-discovers the newest checkpoint and lets you configure the target return (`--target_return`), device (`--device`), and render mode (`--render_mode human/rgb_array`).

## Notes
- **Action head** uses `tanh` to fit MuJoCo action range `[-1, 1]`.
- The trainer and evaluation path reuse your original `training/` and `evaluation/` modules.
- You can switch to other D4RL tasks by changing `--env` / `--dataset` arguments.
- This implementation focuses on *training-time* GQA and RoPE; KV-cache optimizations for inference are not included (not required for DT training).

## Suggested Large Config (≈0.6B)
- `embed_dim=2048`, `n_layer=24`, `n_head=16`, `n_kv_head=8`, `mlp_ratio=5.4`
- Sequence length `K=20` is common in DT; adjust based on memory.
- Mixed precision (`torch.cuda.amp`) can be enabled if desired in the trainer.

## Parameter Count
The total number of parameters decomposes into input embeddings, the transformer core, and output heads. Define:

- `H = embed_dim`
- `N = n_layer`
- `n_h = n_head`
- `n_kv = n_kv_head`
- `S = state_dim`
- `A = act_dim`
- `T = max_episode_len`
- `F = int(mlp_ratio * H)`
- `r = n_kv / n_h`

Then:
```
Params_embed       = H * (T + S + A + 5)
Params_block       = 2H + H^2 * (2 + 2r) + 3 * H * F
Params_transformer = N * Params_block + H
Params_heads       = H*S + S + H*A + A + H + 1

Params_total = Params_embed + Params_transformer + Params_heads
```

For example, the default large model (`H=2048`, `N=24`, `n_h=16`, `n_kv=8`, `mlp_ratio=5.4`, `S≈348`, `A=17`, `T≈1000`) has roughly **1.94B** parameters; the smaller sanity-check setting (`H=768`, `N=12`, `n_h=12`, `n_kv=4`, `mlp_ratio=4.0`, `S≈348`, `A=17`, `T≈1000`) has about **1.05×10⁸** parameters.

Good luck and happy training!
