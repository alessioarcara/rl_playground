from enum import Enum

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str = None,
        size: int = 10,
        reset_success_count: int | None = None,
    ):
        self.size = size
        self.reset_success_count = reset_success_count
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512  # The size of the PyGame window
        self.window = None

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

    def _get_obs(self):
        return {"agent": self._agent_location, "goal": self._goal_location}

    def _get_info(self):
        return {
            "success_count": self.success_count,
        }

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
        if terminated:
            self.success_count += 1
            reward = 0
        else:
            reward = -1
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def _reset_positions(self):
        should_reset = self._agent_location[0] == -1 or (
            self.reset_success_count is not None
            and self.success_count >= self.reset_success_count
        )

        if should_reset:
            self.success_count = 0

            # Sample agent location
            self._agent_location = np.array([0, 0])
            # self._agent_location = self.np_random.integers(
            #     low=0, high=self.size, size=2, dtype=int
            # )
            self._initial_agent_location = self._agent_location.copy()

            # Sample goal location different from agent
            self._goal_location = self._agent_location.copy()
            while np.array_equal(self._goal_location, self._agent_location):
                self._goal_location = self.np_random.integers(
                    low=0, high=self.size, size=2, dtype=int
                )
        else:
            self._agent_location = self._initial_agent_location.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_positions()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target (goal)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._goal_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
