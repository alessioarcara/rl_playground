from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "goal": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self._agent_location = np.array([-1, -1], dtype=int)
        self._goal_location = np.array([-1, -1], dtype=int)
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, -1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, 1]),
        }
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {"agent": self._agent_location, "goal": self._goal_location}

    def _get_info(self):
        return {}

    def step(self, action):
        """
        It accepts an action and return a 5-tuple (observation, reward, terminated, truncated, info)
        """
        # Move agent
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # Check termination
        terminated = np.array_equal(self._agent_location, self._goal_location)
        reward = 0 if terminated else -1
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Sample agent location
        self._agent_location = self.np_random.integers(
            low=0, high=self.size, size=2, dtype=int
        )
        # Sample goal location different from agent
        self._goal_location = self._agent_location.copy()
        while np.array_equal(self._goal_location, self._agent_location):
            self._goal_location = self.np_random.integers(
                low=0, high=self.size, size=2, dtype=int
            )
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self):
        grid = np.full((self.size, self.size), fill_value=".")
        ax, ay = self._agent_location
        gx, gy = self._goal_location
        grid[ay, ax] = "A"
        grid[gy, gx] = "G"
        ascii_grid = "\n".join(" ".join(row) for row in grid)
        return ascii_grid
