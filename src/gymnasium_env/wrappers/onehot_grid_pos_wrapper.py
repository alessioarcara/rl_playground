import gymnasium as gym
import numpy as np
from gym import ObservationWrapper
from gymnasium.spaces import Box


class OneHotGridPosWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, grid_size: int = 10):
        super().__init__(env)
        self.grid_size = grid_size
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(grid_size, grid_size, 2),
            dtype=np.float32,
        )

    def observation(self, obs):
        ag = obs["agent"]
        go = obs["goal"]
        screen = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        screen[ag[1], ag[0], 0] = 1.0
        screen[go[1], go[0], 1] = 1.0
        return screen
