# 第三次实践作业：马里奥场景下的 DQN

## 简介
在这个项目中，我们将实现深度强化学习中的 DQN，并将其应用到 OpenAI Gym 下的马里奥场景中。

本次作业所需的文件为：`Mario-RL.zip`

## 你需要编辑的文件
文件 | 作用
--- | ---
`agent.py` | 用于实现马里奥场景下的 DQN 智能体

## 你可能需要阅读的文件
文件 | 作用
--- | ---
`main.py` | 训练入口
`neural.py` | 定义 Q 网络 `MarioNet`
`wrappers.py` | 环境预处理，包括灰度化、缩放、跳帧、堆叠帧
`metrics.py` | 记录训练过程中的 reward、loss、Q value 等指标
`README.md` | 框架代码运行说明

## 题目：DQN 方法
在本题中，你需要在 `agent.py` 中实现 DQN 所需的核心函数，使得马里奥智能体能够在像素场景下进行训练。

请按如下步骤在 `agent.py` 文件中补全 DQN 智能体：

`td_estimate(state, action)`：根据 online 网络给出的 Q 值，返回当前 batch 中对应动作的 `Q(s,a)`。

```python
def td_estimate(self, state, action):
    """
    Return the TD estimate Q_online(s, a).
    """
    "*** YOUR CODE HERE ***"
```

提示：

- 用 **online** 网络算 Q 值：`current_q_values = self.net(state, model="online")`，得到形状 `[batch_size, action_dim]` 的表（每行一个画面，每列一种按键）。
- `action` 是马里奥实际按的键；取出对应分数：`current_q = current_q_values[np.arange(0, self.batch_size), action]`，再 `return current_q`。

`td_target(reward, next_state, done)`：根据 DQN 的目标公式，返回当前 batch 的 TD target。

```python
@torch.no_grad()
def td_target(self, reward, next_state, done):
    """
    Return the TD target.
    """
    "*** YOUR CODE HERE ***"
```

提示：

- 算训练目标：结束了（`done=True`）只要 `reward`；没结束还要加 `reward + gamma × 下一状态最高分`。
- 用 **target** 网络（不是 online）：`next_q_values = self.net(next_state, model="target")`，再 `next_q = next_q_values.max(dim=1)[0]`。
- `return (reward + (1 - done.float()) * self.gamma * next_q).float()`。头上已有 `@torch.no_grad()`，不用额外写。

`update_Q_online(td_estimate, td_target)`：根据 TD estimate 和 TD target 计算损失，并对 online 网络完成一次参数更新。

```python
def update_Q_online(self, td_estimate, td_target):
    """
    Update the online network once.
    """
    "*** YOUR CODE HERE ***"
```

提示：

- 算 loss，再按 PyTorch 常规流程更新：**loss → zero_grad → backward → step**。
- 依次写：`loss = self.loss_fn(td_estimate, td_target)`，`self.optimizer.zero_grad()`，`loss.backward()`，`self.optimizer.step()`，最后 `return loss.item()`。

当你实现好后，可以使用下面的命令验证 DQN 是否能够运行：

```bash
python main.py --episodes 100 --gpu 0
```

如果有多张 GPU，可以将 `--gpu 0` 改成对应的卡号。

## 实验报告要求
请在实验报告中说明以下内容：

1. 你补全了 `agent.py` 中哪些函数。
2. DQN 的 TD target 是如何计算的。
3. 你的程序是否成功运行，训练输出结果如何。
4. 长周期训练与收敛分析（重点）：
   - 马里奥任务通常需要较长训练周期，建议进行上万 episode 的训练（如 10000+）。
   - 请结合 reward / loss / Q value 曲线，分析训练是否出现收敛趋势。
   - 说明你观察到的阶段性现象（例如前期探索波动、中期提升、后期平台期或不稳定）。

## 代码提交与评分
请你打包提交 `agent.py` 代码文件，并提交一份 `pdf` 或 `markdown` 格式的报告。最终提交的压缩包需要命名为 `"学号_姓名.zip"`。请严格按照如下格式提交。

```text
学号_姓名.zip
│
├── main.py
├── agent.py
├── neural.py
├── wrappers.py
├── metrics.py
└── 报告.pdf / 报告.md
```

评分标准如下：

- 环境运行与代码结构：20 分
- DQN 实现：50 分
- 实验报告：15 分
- 代码规范：5 分
- 拔高部分（长周期训练与收敛分析）：10 分

其中：

- DQN 实现主要考察 `td_estimate`、DQN 的 `td_target` 和参数更新过程
- 拔高部分为重点考察项：马里奥任务通常需要长周期训练，建议至少进行上万 episode（10000+）训练
- 需基于 reward / loss / Q value 的整体趋势判断是否收敛，并在报告中给出阶段性分析与结论
- 只要代码能够正常运行，并完成 DQN 的核心逻辑，即可获得大部分基础分数

## 学术诚信
我们会将你的代码与其他提交进行逻辑查重。请独立完成作业，不要直接拷贝他人代码。
