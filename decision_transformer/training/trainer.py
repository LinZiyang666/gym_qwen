import time
import torch
import numpy as np

class Trainer:
    """
    Minimal, safe base trainer. SequenceTrainer will inherit and override train_step().
    """

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = {}

    def train_step(self):
        """
        Fallback NOP step if child doesn't override.
        """
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = actions.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(None, action_preds, None, None, action_target, None)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, progress_callback=None):
        logs = {}
        train_start = time.time()
        self.model.train()
        total_loss = 0.0
        self.diagnostics = {}

        for step in range(num_steps):
            loss = self.train_step()
            total_loss += float(loss)
            if self.scheduler is not None:
                self.scheduler.step()
            # 如果外部传入了进度回调（例如 tqdm.update），这里安全调用一下
            if progress_callback is not None:
                try:
                    progress_callback(1)  # 默认每步推进 1
                except Exception:
                    pass

        logs['training/train_loss_mean'] = total_loss / max(1, num_steps)
        logs['training/iter_time_sec'] = time.time() - train_start
        logs.update(self.diagnostics)

        # eval
        self.model.eval()
        for i, eval_fn in enumerate(self.eval_fns):
            with torch.no_grad():
                result = eval_fn(self.model)
            if isinstance(result, dict):
                for k, v in result.items():
                    logs[f'evaluation/{k}'] = v
            else:
                logs[f'evaluation/return_mean_{i}'] = result

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                if isinstance(v, (int, float, np.floating)):
                    print(f'{k}: {v:.6f}')
                else:
                    print(f'{k}: {v}')
        return logs
