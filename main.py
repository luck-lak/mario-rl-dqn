#!/usr/bin/env python3
"""
学生版训练入口：DQN。
"""
###设定随机种子代码由claude编写（AI声明1）
from __future__ import annotations

import argparse
import datetime
import random  # [seed-fix] 固定随机种子用
from pathlib import Path

import numpy as np  # [seed-fix] 固定随机种子用
import torch  # [seed-fix] 固定随机种子用

from agent import Mario
from metrics import MetricLogger
from wrappers import make_env


def main():
    parser = argparse.ArgumentParser(description="Train Mario with DQN")
    parser.add_argument("--level", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")  # [seed-fix] 随机种子参数
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint to resume training from")  # [策略C] 支持断点续训
    args = parser.parse_args()

    # [seed-fix] === 固定随机种子，保证可复现性 ===
    SEED = args.seed  # [seed-fix] 从命令行读取种子值
    random.seed(SEED)  # [seed-fix] 固定 Python 标准库 random
    np.random.seed(SEED)  # [seed-fix] 固定 NumPy 随机
    torch.manual_seed(SEED)  # [seed-fix] 固定 PyTorch CPU 随机
    if torch.cuda.is_available():  # [seed-fix]
        torch.cuda.manual_seed_all(SEED)  # [seed-fix] 固定所有 GPU 随机
    torch.backends.cudnn.deterministic = True  # [seed-fix] 强制 CUDA 确定性算法
    torch.backends.cudnn.benchmark = False  # [seed-fix] 禁用自动优化（避免非确定性）
    # [seed-fix] ======================================

    save_dir = args.save_dir or (
        Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.level)
    # env.seed(SEED)  # [seed-fix] 固定 Gym 环境内部随机性
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir,
        checkpoint=args.checkpoint,  # [策略C] 传入 checkpoint 路径，支持断点续训
        gpu_id=args.gpu,
    )
    logger = MetricLogger(save_dir)

    for e in range(args.episodes):
        state = env.reset()

        while True:
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            logger.log_step(reward, loss, q)

            state = next_state
            if done or info.get("flag_get", False):
                break

        logger.log_episode()
        ##修改（1）
        ##开始跑长程记录（2）
        if e % 20 == 0 or e == args.episodes - 1:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step,
            )

    if mario.curr_step > 0:
        mario.save("mario_net_final.chkpt")

    env.close()


if __name__ == "__main__":
    main()
