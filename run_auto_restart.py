#!/usr/bin/env python3
"""
============================================================
【自动重启训练脚本 - 临时添加】
功能：运行 main.py 训练，若异常退出且当前时间在 1:00 之前，自动重启。
用法：python run_auto_restart.py
版本回调：直接删除本文件即可，未修改任何其他代码
============================================================
"""

import subprocess
import sys
import datetime

# === 配置区 ===
# [策略C] 从最新 checkpoint 恢复训练，保持 epsilon 和步数不重置
CHECKPOINT = "checkpoints/2026-05-21T10-05-34/mario_net_1.chkpt"
CMD = [sys.executable, "main.py", "--episodes", "10000", "--gpu", "0",
       "--checkpoint", CHECKPOINT]
RESTART_DEADLINE_HOUR = 1  # 1:00 之前才重启
# ==============


def main():
    while True:
        now = datetime.datetime.now()
        print(f"\n{'='*60}")
        print(f"[{now}] 启动训练: {' '.join(CMD)}")
        print(f"{'='*60}")

        result = subprocess.run(CMD)

        now = datetime.datetime.now()

        # 正常结束
        if result.returncode == 0:
            print(f"\n[{now}] 训练正常结束（退出码 0）。")
            break

        # 异常退出
        print(f"\n[{now}] [警告] 训练异常退出，退出码：{result.returncode}")

        # 检查是否在 1:00 之前
        if now.hour >= RESTART_DEADLINE_HOUR:
            print(f"[{now}] 当前时间已过 {RESTART_DEADLINE_HOUR}:00，不再重启。")
            break

        print(f"[{now}] 当前时间在 {RESTART_DEADLINE_HOUR}:00 之前，准备重启...")


if __name__ == "__main__":
    main()
