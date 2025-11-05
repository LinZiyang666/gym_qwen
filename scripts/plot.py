import argparse
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd


def _normalize_dataset_name(env, dataset_value):
    if dataset_value is None:
        return None

    dataset = dataset_value.split('/')[-1]
    if env == 'humanoid' and dataset.endswith('-v0'):
        dataset = dataset[:-3]
    return dataset


def _get_env_d4rl_name(args):
    if args.env_d4rl_name:
        return args.env_d4rl_name

    env = args.env.lower() if args.env else None
    dataset = _normalize_dataset_name(env, args.dataset)

    if env is None:
        raise ValueError("Provide --env/--dataset or --env_d4rl_name so the log prefix can be resolved.")

    if env == 'humanoid':
        return f'humanoid-{dataset}-v5'

    supported_envs = {'walker2d', 'halfcheetah', 'hopper'}
    if env not in supported_envs:
        raise ValueError(f"Unsupported env '{env}'. Expected one of {sorted(supported_envs | {'humanoid'})}.")

    return f'{env}-{dataset}-v2'


def plot(args):

    env_d4rl_name = _get_env_d4rl_name(args)
    log_dir = args.log_dir
    x_key = args.x_key
    if args.y_key:
        y_key = args.y_key
    elif env_d4rl_name.startswith('humanoid-'):
        y_key = 'eval_avg_reward'
    else:
        y_key = 'eval_d4rl_score'
    y_smoothing_win = args.smoothing_window
    plot_avg = args.plot_avg
    save_fig = args.save_fig

    if plot_avg:
        save_fig_path = env_d4rl_name + "_avg.png"
    else:
        save_fig_path = env_d4rl_name + ".png"

    all_files = []
    for prefix in ('dt', 'dtqwen3'):
        pattern = os.path.join(log_dir, f'{prefix}_{env_d4rl_name}*.csv')
        all_files.extend(glob.glob(pattern))
    all_files = sorted(all_files)

    if not all_files:
        raise FileNotFoundError(
            f"No log files matched any of {[f'{p}_{env_d4rl_name}*.csv' for p in ('dt', 'dtqwen3')]} under '{log_dir}'."
        )

    ax = plt.gca()
    ax.set_title(env_d4rl_name)

    if plot_avg:
        name_list = []
        df_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            df_list.append(frame)

        df_concat = pd.concat(df_list)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        data_avg.plot(x=x_key, y='y_smooth', ax=ax)

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(['avg of all runs'], loc='lower right')

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()

    else:
        name_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            frame.plot(x=x_key, y='y_smooth', ax=ax)
            name_list.append(filename.split('/')[-1])

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(name_list, loc='lower right')

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_d4rl_name', type=str, default=None,
                        help="Full D4RL-style environment identifier, e.g. halfcheetah-medium-v2")
    parser.add_argument('--env', type=str, default='halfcheetah',
                        help="Short environment key shared with train.py (walker2d, halfcheetah, hopper, humanoid).")
    parser.add_argument('--dataset', type=str, default='medium',
                        help="Dataset split; for Humanoid you can also pass Minari ids like mujoco/humanoid/medium-v0.")
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--x_key', type=str, default='num_updates')
    parser.add_argument('--y_key', type=str, default=None,
                        help="Metric to plot on Y axis; default is eval_d4rl_score except Humanoid uses eval_avg_reward.")
    parser.add_argument('--smoothing_window', type=int, default=1)
    parser.add_argument("--plot_avg", action="store_true", default=False,
                    help="plot avg of all logs else plot separately")
    parser.add_argument("--save_fig", action="store_true", default=False,
                    help="save figure if true")

    args = parser.parse_args()

    plot(args)
