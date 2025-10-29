#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download/convert MuJoCo locomotion datasets to D4RL-style pickles.

Priority:
  1) Minari (mujoco/<env>/{simple|medium|expert}-v0)
  2) Fallback to d4rl for 'medium-replay' and legacy '-v2'

Outputs:
  <env>-<dataset>-v2.pkl  # paths list with keys:
    observations, next_observations, actions, rewards, terminals
"""

import argparse
import os
import collections
import pickle
import numpy as np
import gymnasium as gym
import gym as ogym  # D4RL registers envs on OpenAI Gym (not Gymnasium)

# ---------- Minari availability ----------
try:
    import minari  # noqa: F401
    HAS_MINARI = True
    MINARI_ERR = None
except Exception as e:
    HAS_MINARI = False
    MINARI_ERR = e


def check_minari_runtime():
    """Return (ok, msg) — ok=True only if minari + h5py + pillow are usable."""
    if not HAS_MINARI:
        return False, f"minari not importable: {MINARI_ERR}"
    missing = []
    try:
        import h5py  # noqa: F401
    except Exception:
        missing.append('h5py (`pip install "minari[hdf5]"` 或 `conda install h5py`)')
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        missing.append("Pillow (`pip install pillow`)")
    if missing:
        return False, " 和 ".join(missing)
    return True, ""


def minari_id(env_name: str, dataset_type: str) -> str:
    # Minari locomotion: mujoco/<env>/{simple|medium|expert}-v0
    return f"mujoco/{env_name}/{dataset_type}-v0"


def d4rl_env_id(env_name: str, dataset_type: str) -> str:
    # Base D4RL naming (no version suffix here): <env>-<dataset>
    # We'll try -v2 / bare / -v0 later.
    return f"{env_name}-{dataset_type}"


def ensure_minari_download(mid: str, force: bool = False, verbose: bool = True):
    """Use minari.download_dataset to fetch from remote."""
    import minari as _m
    if verbose:
        print(f"[Minari] downloading dataset: {mid} (force={force})")
    _m.download_dataset(mid, force_download=force)


def build_paths_from_minari_dataset(ds):
    """Convert Minari dataset to D4RL-style paths with next_observations."""
    paths = []
    for ep in ds.iterate_episodes():
        obs = ep.observations.astype(np.float32)
        acts = ep.actions.astype(np.float32)
        rews = ep.rewards.astype(np.float32).reshape(-1)
        terms = ep.terminations.astype(bool).reshape(-1)  # truncations 不计入 terminals

        # 对齐：常见 len(obs)=len(act)+1；也兼容 len(obs)=len(act)
        if obs.shape[0] == acts.shape[0] + 1:
            next_obs = obs[1:].copy()
            obs = obs[:-1].copy()
        elif obs.shape[0] == acts.shape[0]:
            if obs.shape[0] > 1:
                next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0).copy()
            else:
                next_obs = obs.copy()
        else:
            raise ValueError(f"Minari episode shapes not aligned: obs {obs.shape}, actions {acts.shape}")

        paths.append({
            "observations": obs,
            "next_observations": next_obs,
            "actions": acts,
            "rewards": rews,
            "terminals": terms,
        })
    return paths


def build_paths_from_d4rl_dataset(dataset, horizon_default=1000):
    """Old d4rl stitching with timeouts handling."""
    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    use_timeouts = "timeouts" in dataset
    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        final_timestep = bool(dataset["timeouts"][i]) if use_timeouts else (episode_step == horizon_default - 1)

        for k in ["observations", "next_observations", "actions", "rewards", "terminals"]:
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_data = {k: np.array(v) for k, v in data_.items()}
            paths.append(episode_data)
            data_.clear()
            data_ = collections.defaultdict(list)
            episode_step = 0
        else:
            episode_step += 1
    return paths


def summarize_and_dump(paths, out_pkl: str):
    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = int(np.sum([p["rewards"].shape[0] for p in paths]))
    print(f"Number of episodes: {len(paths)}")
    print(f"Number of samples collected: {num_samples}")
    print(f"Trajectory returns: mean={np.mean(returns):.3f}, std={np.std(returns):.3f}, "
          f"max={np.max(returns):.3f}, min={np.min(returns):.3f}")
    os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(paths, f)
    print(f"Saved: {out_pkl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="all",
                    help="halfcheetah|hopper|walker2d|humanoid|all")
    ap.add_argument("--datasets", type=str, default="medium,expert,medium-replay",
                    help="comma-separated: simple,medium,expert,medium-replay")
    ap.add_argument("--out", type=str, default="data",
                    help="kept for compatibility; pkl files are saved in CWD")
    ap.add_argument("--force-download-minari", action="store_true",
                    help="call minari.download_dataset before load_dataset")
    args = ap.parse_args()

    envs = ["halfcheetah", "hopper", "walker2d", "humanoid"] if args.env == "all" else [args.env]
    ds_list = [s.strip() for s in args.datasets.split(",") if s.strip()]

    ok_minari, minari_msg = check_minari_runtime()

    # D4RL has replay datasets only for these MuJoCo envs
    HAS_REPLAY = {"halfcheetah", "hopper", "walker2d", "ant"}

    for env_name in envs:
        for dataset_type in ds_list:
            print("=" * 80)
            print(f"[Target] {env_name}-{dataset_type}")
            out_pkl = f"{env_name}-{dataset_type}-v2.pkl"
            paths = None

            # ----- Minari first for {simple, medium, expert}
            if dataset_type in {"simple", "medium", "expert"} and ok_minari:
                try:
                    mid = minari_id(env_name, dataset_type)
                    if args.force_download_minari:
                        ensure_minari_download(mid, force=False)
                    print(f"[Minari] loading {mid} ...")
                    import minari
                    try:
                        ds = minari.load_dataset(mid, download=True)  # auto-download if needed
                    except Exception as e1:
                        msg = str(e1)
                        # Force re-download if cache is incomplete
                        if "No data found in data path" in msg or "not found" in msg.lower():
                            print(f"[Minari] cache incomplete, force re-download: {mid}")
                            ensure_minari_download(mid, force=True)
                            ds = minari.load_dataset(mid, download=False)
                        else:
                            raise
                    paths = build_paths_from_minari_dataset(ds)
                except Exception as e:
                    print(f"[Minari] failed: {e}. Will try d4rl fallback.")

            elif dataset_type in {"simple", "medium", "expert"} and not ok_minari:
                print(f"[Minari] skipped: {minari_msg}. Trying d4rl fallback (if available).")

            # ----- d4rl fallback (needed for medium-replay / legacy -v2 / Minari miss)
            if paths is None:
                try:
                    import d4rl  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "Minari 不可用或该数据集缺失，且未安装 d4rl。\n"
                        "Minari（推荐）：\n  pip install 'minari[hdf5]' pillow imageio\n"
                        "d4rl 回退（用于 medium-replay / -v2）：\n"
                        '  pip install "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"\n'
                        "注意：d4rl locomotion 需要 mujoco-py 与 ~/.mujoco/mujoco210。\n"
                        f"(debug: d4rl import error: {e})"
                    )
                try:
                    if dataset_type == "medium-replay" and env_name not in HAS_REPLAY:
                        print(f"[skip] {env_name}-medium-replay not available in D4RL/Minari; skipping.")
                        paths = []
                    else:
                        eid = d4rl_env_id(env_name, dataset_type)  # e.g., "halfcheetah-medium-replay"
                        print(f"[d4rl] loading {eid} via env.get_dataset() ...")

                        # Try common D4RL variants: v2 (most MuJoCo tasks), then bare, then v0
                        candidates = [f"{eid}-v2", eid, f"{eid}-v0"]
                        last_exc = None
                        env = None
                        for cid in candidates:
                            try:
                                print(f"[d4rl] trying env id: {cid}")
                                env = ogym.make(cid)  # OpenAI Gym registry
                                print(f"[d4rl] using env id: {cid}")
                                break
                            except Exception as _e:
                                last_exc = _e

                        if env is None:
                            raise RuntimeError(
                                f"None of these env IDs exist for {eid}: {candidates} (last error: {last_exc})"
                            )

                        dataset = env.get_dataset()
                        paths = build_paths_from_d4rl_dataset(dataset)

                except Exception as e:
                    raise RuntimeError(f"Failed to load {env_name}-{dataset_type} from d4rl: {e}")

            # Skip saving when no data was found/available
            if not paths:
                print(f"[skip] no dataset produced for {env_name}-{dataset_type}.")
                continue

            summarize_and_dump(paths, out_pkl)


if __name__ == "__main__":
    main()