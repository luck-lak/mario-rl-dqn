import gym
import numpy as np
from gym.spaces import Box

from compat import patch_runtime_compat


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env, keep_dim=False):
        super().__init__(env)
        self.keep_dim = keep_dim

        obs_shape = self.observation_space.shape[:2]
        if keep_dim:
            obs_shape = obs_shape + (1,)

        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        # Use a lightweight RGB -> grayscale conversion to reduce CPU overhead.
        observation = observation.astype(np.float32, copy=False)
        gray = (
            0.299 * observation[..., 0]
            + 0.587 * observation[..., 1]
            + 0.114 * observation[..., 2]
        ).astype(np.uint8)
        if self.keep_dim:
            gray = np.expand_dims(gray, axis=-1)
        return gray


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        input_shape = self.observation_space.shape
        self._src_height = input_shape[0]
        self._src_width = input_shape[1]
        self._row_idx = np.linspace(0, self._src_height - 1, self.shape[0]).astype(np.int32)
        self._col_idx = np.linspace(0, self._src_width - 1, self.shape[1]).astype(np.int32)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        # Nearest-neighbor downsampling is much cheaper than generic resize.
        return observation[self._row_idx][:, self._col_idx]


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return observation.astype(np.float32, copy=False) / 255.0


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def make_env(level="SuperMarioBros-1-1-v0"):
    patch_runtime_compat()

    import gym_super_mario_bros
    from gym.wrappers import FrameStack
    from nes_py.wrappers import JoypadSpace

    env = gym_super_mario_bros.make(level)
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = NormalizeObservation(env)
    env = FrameStack(env, num_stack=4)
    return env
