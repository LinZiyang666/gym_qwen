#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minari-based dataset manager for Humanoid-v5 training data.

This script downloads and manages offline RL datasets for Humanoid-v5 using Minari,
which provides clean, versioned datasets from the Farama Foundation.

Supported datasets:
  - simple: Basic demonstrations (lower performance)
  - medium: Medium-quality expert data
  - expert: High-quality expert demonstrations

Usage examples:
  # Download all Humanoid datasets
  python download_humanoid_v5_minari.py --datasets simple,medium,expert

  # Download only medium dataset
  python download_humanoid_v5_minari.py --datasets medium

  # Download and export to pickle format for compatibility
  python download_humanoid_v5_minari.py --datasets medium --export-pkl

  # Force re-download even if cached
  python download_humanoid_v5_minari.py --datasets medium --force

Output:
  - Minari datasets are cached in: ~/.minari/datasets/
  - Optional pickle exports: <dataset>-v5.pkl (if --export-pkl is used)
    Format: List of trajectory dicts with keys:
      observations, next_observations, actions, rewards, terminals
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def check_minari_environment():
    """
    Verify that minari and its dependencies are properly installed.

    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        import minari  # noqa: F401
    except ImportError as e:
        return False, f"minari not installed: {e}\n  Install: pip install 'minari[hdf5]'"

    missing = []
    try:
        import h5py  # noqa: F401
    except ImportError:
        missing.append("h5py (install: pip install 'minari[hdf5]' or conda install h5py)")

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow (install: pip install pillow)")

    if missing:
        return False, "Missing dependencies:\n  " + "\n  ".join(missing)

    return True, ""


def get_minari_dataset_id(dataset_type: str) -> str:
    """
    Get the Minari dataset ID for Humanoid.

    Args:
        dataset_type: One of 'simple', 'medium', 'expert'

    Returns:
        str: Minari dataset ID (e.g., 'mujoco/humanoid/medium-v0')
    """
    return f"mujoco/humanoid/{dataset_type}-v0"


def download_dataset(dataset_type: str, force: bool = False, verbose: bool = True):
    """
    Download a Humanoid dataset from Minari.

    Args:
        dataset_type: Dataset quality level ('simple', 'medium', 'expert')
        force: Force re-download even if cached locally
        verbose: Print progress messages
    """
    import minari

    dataset_id = get_minari_dataset_id(dataset_type)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_id}")
        print(f"{'=' * 70}")

    try:
        if force or verbose:
            print(f"Downloading {dataset_id}...")
            minari.download_dataset(dataset_id, force_download=force)

        # Verify by loading
        if verbose:
            print(f"Loading {dataset_id} to verify...")
        ds = minari.load_dataset(dataset_id, download=not force)

        # Print dataset statistics
        total_episodes = ds.total_episodes
        total_steps = ds.total_steps

        if verbose:
            print(f"\nDataset loaded successfully:")
            print(f"  Total episodes: {total_episodes}")
            print(f"  Total steps: {total_steps}")

            # Compute return statistics
            returns = []
            for ep in ds.iterate_episodes():
                returns.append(float(np.sum(ep.rewards)))

            returns = np.array(returns)
            print(f"\nReturn statistics:")
            print(f"  Mean: {np.mean(returns):.2f}")
            print(f"  Std:  {np.std(returns):.2f}")
            print(f"  Min:  {np.min(returns):.2f}")
            print(f"  Max:  {np.max(returns):.2f}")
            print(f"  Median: {np.percentile(returns, 50):.2f}")
            print(f"  P90:    {np.percentile(returns, 90):.2f}")

        return ds

    except Exception as e:
        print(f"Error downloading/loading {dataset_id}: {e}")
        raise


def convert_minari_to_paths(dataset) -> List[Dict[str, np.ndarray]]:
    """
    Convert Minari dataset to D4RL-style paths format.

    This format is compatible with the existing experiment_qwen.py script.

    Args:
        dataset: Minari dataset object

    Returns:
        List of trajectory dictionaries with keys:
          - observations: (T, obs_dim)
          - next_observations: (T, obs_dim)
          - actions: (T, act_dim)
          - rewards: (T,)
          - terminals: (T,)
    """
    paths = []

    for ep in dataset.iterate_episodes():
        obs = ep.observations.astype(np.float32)
        acts = ep.actions.astype(np.float32)
        rews = ep.rewards.astype(np.float32).reshape(-1)
        terms = ep.terminations.astype(bool).reshape(-1)

        # Handle observation/action alignment
        # Minari typically has len(obs) = len(actions) + 1 (includes final obs)
        if obs.shape[0] == acts.shape[0] + 1:
            next_obs = obs[1:].copy()
            obs = obs[:-1].copy()
        elif obs.shape[0] == acts.shape[0]:
            # If already aligned, construct next_obs
            if obs.shape[0] > 1:
                next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0).copy()
            else:
                next_obs = obs.copy()
        else:
            raise ValueError(
                f"Unexpected episode shapes: obs {obs.shape}, actions {acts.shape}"
            )

        paths.append({
            "observations": obs,
            "next_observations": next_obs,
            "actions": acts,
            "rewards": rews,
            "terminals": terms,
        })

    return paths


def export_to_pickle(paths: List[Dict[str, np.ndarray]],
                    output_path: str,
                    verbose: bool = True):
    """
    Export paths to pickle file.

    Args:
        paths: List of trajectory dictionaries
        output_path: Output pickle file path
        verbose: Print progress messages
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Compute statistics
    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = int(np.sum([p["rewards"].shape[0] for p in paths]))

    if verbose:
        print(f"\nExporting to pickle: {output_path}")
        print(f"  Episodes: {len(paths)}")
        print(f"  Timesteps: {num_samples}")
        print(f"  Avg return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

    with open(output_path, "wb") as f:
        pickle.dump(paths, f)

    if verbose:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"Saved successfully!")


def list_downloaded_datasets(verbose: bool = True):
    """List all locally cached Minari datasets for Humanoid."""
    try:
        import minari

        if verbose:
            print("\nChecking for locally cached Humanoid datasets...")

        all_local = minari.list_local_datasets()
        humanoid_datasets = [d for d in all_local if 'humanoid' in d.lower()]

        if humanoid_datasets:
            if verbose:
                print(f"Found {len(humanoid_datasets)} Humanoid dataset(s):")
                for ds_id in humanoid_datasets:
                    print(f"  - {ds_id}")
        else:
            if verbose:
                print("No Humanoid datasets found locally.")

        return humanoid_datasets

    except Exception as e:
        if verbose:
            print(f"Error listing datasets: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Download and manage Humanoid-v5 datasets using Minari",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all dataset types
  python download_humanoid_v5_minari.py --datasets simple,medium,expert

  # Download medium dataset and export to pickle
  python download_humanoid_v5_minari.py --datasets medium --export-pkl

  # List cached datasets
  python download_humanoid_v5_minari.py --list

  # Force re-download
  python download_humanoid_v5_minari.py --datasets medium --force

Note: Minari datasets are cached in ~/.minari/datasets/ by default.
      The experiment_qwen.py script can load directly from Minari,
      so pickle export is optional (mainly for compatibility/inspection).
        """
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="medium",
        help="Comma-separated list of datasets to download: simple,medium,expert (default: medium)"
    )

    parser.add_argument(
        "--export-pkl",
        action="store_true",
        help="Export datasets to pickle format for compatibility with older scripts"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for pickle files (default: current directory)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset is cached locally"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all locally cached Humanoid datasets and exit"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Check environment
    if verbose:
        print("Humanoid-v5 Dataset Manager (Minari)")
        print("=" * 70)

    success, error_msg = check_minari_environment()
    if not success:
        print(f"\nError: {error_msg}")
        return 1

    if verbose:
        print("Environment check: OK")
        try:
            import minari
            print(f"Minari version: {minari.__version__}")
        except:
            pass

    # List mode
    if args.list:
        list_downloaded_datasets(verbose=verbose)
        return 0

    # Parse dataset types
    dataset_types = [s.strip() for s in args.datasets.split(",") if s.strip()]
    valid_types = {"simple", "medium", "expert"}

    for ds_type in dataset_types:
        if ds_type not in valid_types:
            print(f"Error: Invalid dataset type '{ds_type}'. Must be one of: {valid_types}")
            return 1

    if not dataset_types:
        print("Error: No datasets specified")
        return 1

    # Download and optionally export each dataset
    for ds_type in dataset_types:
        try:
            # Download from Minari
            dataset = download_dataset(ds_type, force=args.force, verbose=verbose)

            # Optionally export to pickle
            if args.export_pkl:
                paths = convert_minari_to_paths(dataset)
                output_filename = f"humanoid-{ds_type}-v5.pkl"
                output_path = os.path.join(args.output_dir, output_filename)
                export_to_pickle(paths, output_path, verbose=verbose)

        except Exception as e:
            print(f"Failed to process dataset '{ds_type}': {e}")
            return 1

    if verbose:
        print("\n" + "=" * 70)
        print("All datasets processed successfully!")
        print("=" * 70)
        if not args.export_pkl:
            print("\nNote: Datasets are cached in Minari's local storage.")
            print("      The experiment_qwen.py script can load them directly.")
            print("      Use --export-pkl if you need pickle files for compatibility.")

    return 0


if __name__ == "__main__":
    exit(main())
