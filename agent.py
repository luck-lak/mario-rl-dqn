"""
学生作业文件：在马里奥场景中实现 DQN。

你需要补全的核心函数：
- `td_estimate`
- `td_target`
- `update_Q_online`
"""

from pathlib import Path
import random

import numpy as np
import torch

from neural import MarioNet


# ============================================================
# 【策略C优化】ReplayBuffer：用 list 实现环形缓冲区
# 替换原版 deque，random.sample 从 O(n) 降到 O(1)
# 支持 clear() 用于 checkpoint 后清空经验池
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.position = 0
        self.size = 0

    def push(self, item):
        self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < batch_size:
            batch_size = self.size
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[i] for i in indices]

    def clear(self):
        """清空经验池，用于 checkpoint 后重新积累高质量数据。"""
        self.buffer = [None] * self.capacity
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size


class Mario:
    def __init__(
        self,
        state_dim,
        action_dim,
        save_dir: Path,
        checkpoint=None,
        gpu_id=0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        if gpu_id is not None and torch.cuda.is_available():
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(f"Invalid gpu_id={gpu_id}, available GPUs: {torch.cuda.device_count()}")
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.use_cuda = self.device.type == "cuda"
        print(f"[Device] Using: {self.device}")

        # 【策略C优化】deque(maxlen=100000) → ReplayBuffer(100000)
        # 解决 random.sample(deque) O(n) 遍历瓶颈
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 64

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975###0.9999885
        ###rate_decay针对1000轮作了适配性调整
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 10000
        self.learn_every = 3
        self.sync_every = 10000
        # 【策略C优化】save_every 从 500000 改到 100000
        # 更频繁保存 checkpoint，同时更频繁清空经验池
        self.save_every = 100000

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device=self.device)
        if checkpoint is not None:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = np.asarray(state, dtype=np.float32)
            state = torch.as_tensor(state).unsqueeze(0)
            if self.use_cuda:
                state = state.to(self.device)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32))
        next_state = torch.as_tensor(np.asarray(next_state, dtype=np.float32))
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)

        # [replay-on-cpu] 回放池数据存在 CPU 内存，避免 GPU 显存溢出
        # 原代码在此处将数据搬到 GPU，导致 6 万条经验占满显存触发 OOM
        # if self.use_cuda:
        #     state = state.to(self.device)
        #     next_state = next_state.to(self.device)
        #     action = action.to(self.device)
        #     reward = reward.to(self.device)
        #     done = done.to(self.device)

        # 【策略C优化】deque.append → buffer.push
        self.memory.push((state, next_state, action, reward, done))

    def recall(self):
        # 【策略C优化】random.sample(deque) → buffer.sample()
        # 从 O(n) 降到 O(1)，解决训练随经验池增大而变慢的核心瓶颈
        batch = self.memory.sample(self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        # [replay-on-cpu] 采样后再统一搬到 GPU
        if self.use_cuda:
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        根据 online 网络返回当前 batch 的 Q(s, a)。

        提示：online 网络算 Q 值表，再按 action 取对应分数。见 ASSIGNMENT.md。
        """
        ##raise NotImplementedError("TODO: implement td_estimate")
        current_q_values = self.net(state, model="online")
        # [策略C修复] 用 state.shape[0] 取实际 batch size，避免经验池数据不足时索引越界
        batch_size_actual = state.shape[0]
        current_q = current_q_values[np.arange(0, batch_size_actual), action]
        return current_q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        根据 DQN 目标公式计算 TD target。

        提示：target 网络算 next_state 最高分，再写 return 公式。见 ASSIGNMENT.md。
        """
        #raise NotImplementedError("TODO: implement td_target")
        next_q_values = self.net(next_state, model="target")
        next_q = next_q_values.max(dim=1)[0]
        return (reward + (1-done.float())*self.gamma*next_q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        使用 `self.loss_fn`、`self.optimizer` 完成一次参数更新。

        提示：loss_fn → zero_grad → backward → step → return loss.item()。
        """
        #raise NotImplementedError("TODO: implement update_Q_online")
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0 and self.curr_step > 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # [策略C修复] checkpoint 清空经验池后，数据不足 batch_size 时跳过学习
        if len(self.memory) < self.batch_size:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    def save(self, save_name=None):
        if save_name is None:
            save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        else:
            save_path = self.save_dir / save_name

        # 【策略C优化】保存 curr_step，支持断点续训时恢复进度
        torch.save(
            {
                "model": self.net.state_dict(),
                "exploration_rate": self.exploration_rate,
                "curr_step": self.curr_step,
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")

        # 【策略C优化】保存 checkpoint 后清空经验池
        # 避免旧策略产生的低质量数据长期累积，让新策略的经验更快主导学习
        self.memory.clear()
        print("[策略C] 经验池已清空，重新开始积累高质量数据...")
        return save_path

    def load(self, load_path):
        checkpoint = torch.load(
            load_path,
            map_location=self.device,
            weights_only=True,
        )
        self.net.load_state_dict(checkpoint["model"])
        # 【策略C优化】兼容旧格式 checkpoint（旧版未保存 curr_step）
        self.exploration_rate = checkpoint.get("exploration_rate", self.exploration_rate)
        self.curr_step = checkpoint.get("curr_step", 0)
        print(f"Loaded checkpoint from {load_path}, curr_step={self.curr_step}, epsilon={self.exploration_rate:.4f}")
