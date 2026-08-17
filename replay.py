#!/usr/bin/env python3
"""
加载 checkpoint 并回放马里奥。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent import Mario
from wrappers import make_env


def main():
    parser = argparse.ArgumentParser(description="Replay a trained Mario checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--level", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        available = sorted(args.checkpoint.parent.glob("*.chkpt"))
        available_text = "\n".join(f"- {path}" for path in available) or "- (none found)"
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Available checkpoints in {args.checkpoint.parent}:\n{available_text}"
        )

    env = make_env(args.level)
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=args.checkpoint.parent,
        checkpoint=args.checkpoint,
        gpu_id=args.gpu,
    )
    mario.exploration_rate = 0.0

    for ep in range(args.episodes):
        state = env.reset()
        total_reward = 0.0
        while True:
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            if done or info.get("flag_get", False):
                print(f"Episode {ep + 1}: reward={total_reward:.1f}, flag_get={info.get('flag_get', False)}")
                break

    env.close()


if __name__ == "__main__":
    main()
