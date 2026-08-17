# Mario DQN Starter Code

该框架目标是在 `gym-super-mario-bros` 上补全并跑通 DQN。

这是马里奥像素场景更常见、更合理的强化学习做法。

## 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate mario-dqn
```

## 2. File Structure

- `main.py`：训练入口
- `agent.py`：需要补全的 DQN 核心逻辑
- `neural.py`：CNN 网络定义
- `wrappers.py`：环境预处理
- `compat.py`：老版本依赖的运行时兼容补丁
- `metrics.py`：训练日志与曲线
- `replay.py`：加载 checkpoint 回放

## 3. What To Read

主要阅读：

- `agent.py`
- `neural.py`
- `wrappers.py`
- `main.py`

其中 `agent.py` 中已经标出 `TODO`，需要自行补全。

## 4. Run

```bash
python main.py --episodes 100 --gpu 0
```

这里的 `100` 只是用于快速检查代码是否能够正常运行的短跑示例。
如果有多张 GPU，可以将 `--gpu 0` 改成对应的卡号。

按照当前默认超参数，训练过程本身是偏长周期设计的。想看到比较明显的训练效果，通常需要运行很长时间；在 episode 数上，往往需要几万甚至更多。从 step 数角度看，也通常对应几十万到上百万步的训练过程。

如果希望更快看到变化，可以自行调整超参数，例如探索率衰减速度、burn-in 长度、target network 同步频率等。

## 5. Replay A Checkpoint

```bash
python replay.py --checkpoint checkpoints/your_run/mario_net_final.chkpt --gpu 0
```

## 6. Notes

- 重点是理解 DQN 的训练流程，不要求必须通关
