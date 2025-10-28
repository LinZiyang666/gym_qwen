# -*- coding: utf-8 -*-
import numpy as np
import torch
from typing import Tuple, Optional

@torch.no_grad()
def evaluate_episode_rtg(
    env_name: str,
    model,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    target_return: float,
    device: str = "cuda",
    rtg_scale: float = 1000.0,
    K: int = 20,
    max_ep_len: int = 1000,
    render: bool = False,
    mode: str = "delayed",
    render_mode: Optional[str] = None,
) -> Tuple[float, int]:
    """
    Rollout with returns-to-go conditioning (Decision Transformer style).
    - Keeps a sliding window of length K.
    - Causal: at step t, predict a_t using info up to (R_t, s_t, a_{t-1}, ...).
    - 'delayed' mode: RTG does NOT decrease over time (as in the original DT paper).
      Otherwise (mode != 'delayed'), RTG will be decreased by observed rewards.

    Returns:
        (episode_return, episode_length)
    """
    import gymnasium as gym

    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_name, **env_kwargs)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, _ = reset_out
    else:
        obs = reset_out
    obs = obs.astype(np.float32)

    state_mean = state_mean.astype(np.float32)
    state_std = state_std.astype(np.float32)

    state_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]

    # Sliding buffers (1, K, *)
    states = torch.zeros(1, K, state_dim, dtype=torch.float32, device=device)
    actions = torch.zeros(1, K, act_dim, dtype=torch.float32, device=device)
    rewards = torch.zeros(1, K, 1, dtype=torch.float32, device=device)
    rtgs = torch.zeros(1, K, 1, dtype=torch.float32, device=device)
    timesteps = torch.zeros(1, K, dtype=torch.long, device=device)
    attn_mask = torch.zeros(1, K, dtype=torch.float32, device=device)

    # Initialize last slot with current obs & target RTG
    cur_rtg = float(target_return) / float(rtg_scale)
    states[0, -1] = torch.from_numpy((obs - state_mean) / state_std).to(device)
    rtgs[0, -1, 0] = cur_rtg
    timesteps[0, -1] = 0
    attn_mask[0, -1] = 1.0

    ep_return = 0.0
    t = 0
    done = False

    model.eval()

    while (not done) and (t < max_ep_len):
        # 预测动作（使用 autocast 有 GPU 则更快）
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.startswith("cuda"))):
            out = model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=rtgs,
                timesteps=timesteps,
                attention_mask=attn_mask,
            )
            # 兼容多种 forward 返回风格
            if isinstance(out, (tuple, list)):
                act_pred = out[1] if len(out) > 1 else out[0]
            elif hasattr(out, "action_preds"):
                act_pred = out.action_preds
            else:
                act_pred = out

            act = act_pred[0, -1].float().detach().cpu().numpy()

        # clip 到动作空间
        act = np.clip(act, env.action_space.low, env.action_space.high)

        # 与环境交互一步
        obs, reward, terminated, truncated, _ = env.step(act)
        obs = obs.astype(np.float32)
        done = bool(terminated or truncated)
        ep_return += float(reward)
        t += 1
        if render:
            env.render()

        # 记录当前动作与奖励，供后续时刻使用
        actions[0, -1] = torch.as_tensor(act, dtype=actions.dtype, device=device)
        rewards[0, -1, 0] = float(reward)

        # 更新 RTG
        if mode != "delayed":
            cur_rtg = max(0.0, cur_rtg - reward / float(rtg_scale))

        # Slide left by 1, then fill last slot with new data
        states = torch.roll(states, shifts=-1, dims=1)
        actions = torch.roll(actions, shifts=-1, dims=1)
        rewards = torch.roll(rewards, shifts=-1, dims=1)
        rtgs = torch.roll(rtgs, shifts=-1, dims=1)
        timesteps = torch.roll(timesteps, shifts=-1, dims=1)
        attn_mask = torch.roll(attn_mask, shifts=-1, dims=1)

        states[0, -1] = torch.from_numpy((obs - state_mean) / state_std).to(device)
        actions[0, -1].zero_()  # 占位；下一次预测会用到 a_{t}
        rewards[0, -1, 0] = 0.0
        rtgs[0, -1, 0] = cur_rtg
        timesteps[0, -1] = t
        attn_mask[0, -1] = 1.0

    env.close()
    return ep_return, t


@torch.no_grad()
def evaluate_episode(
    env,
    state_dim: int,
    act_dim: int,
    model,
    max_ep_len: int = 1000,
    device: str = "cuda",
    target_return: Optional[float] = None,
    mode: str = "normal",
    state_mean: float = 0.0,
    state_std: float = 1.0,
) -> Tuple[float, int]:
    """
    兼容保留：不带显式 RTG 的简单评测（多数场景请用 evaluate_episode_rtg）
    """
    model.eval()
    model.to(device=device)

    state_mean_t = torch.as_tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.as_tensor(state_std, dtype=torch.float32, device=device)

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        state, _ = reset_out
    else:
        state = reset_out
    state = np.asarray(state, dtype=np.float32)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor([0], device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0.0, 0
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.startswith("cuda"))):
            action = model.get_action(
                ((states.to(dtype=torch.float32) - state_mean_t) / state_std_t),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                torch.zeros((1, states.shape[0] + 1, 1), device=device),  # dummy rtg
                timesteps.to(dtype=torch.long),
            )
        actions[-1] = action
        action_np = action.detach().cpu().numpy()

        step_out = env.step(action_np)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            state, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            state, reward, done, _ = step_out

        state = np.asarray(state, dtype=np.float32)
        states = torch.cat([states, torch.from_numpy(state).to(device).reshape(1, state_dim)], dim=0)
        rewards[-1] = reward
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += float(reward)
        episode_length += 1
        if done:
            break

    return float(episode_return), int(episode_length)
