import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class MetricLogger:
    def __init__(self, save_dir: Path, moving_avg_window: int = 100):
        self.save_log = save_dir / "log.txt"
        self.moving_avg_window = moving_avg_window
        self.plot_every_records = 5
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>10}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-self.moving_avg_window :]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-self.moving_avg_window :]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-self.moving_avg_window :]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-self.moving_avg_window :]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_delta = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - Step {step} - Epsilon {epsilon:.4f} - "
            f"Mean Reward {mean_ep_reward} - Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - Mean Q Value {mean_ep_q}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:10d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_delta:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        record_count = len(self.moving_avg_ep_rewards)
        if record_count > 1 and record_count % self.plot_every_records != 0:
            return

        mapping = {
            "ep_rewards": self.ep_rewards_plot,
            "ep_lengths": self.ep_lengths_plot,
            "ep_avg_losses": self.ep_avg_losses_plot,
            "ep_avg_qs": self.ep_avg_qs_plot,
        }
        for metric, plot_path in mapping.items():
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(plot_path)
