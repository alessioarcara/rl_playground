import gymnasium as gym
import numpy as np
from gym import ObservationWrapper
from gymnasium.spaces import Box


def onehot_agent_goal_positions(
    agent: tuple[int, int], goal: tuple[int, int], grid_size: int = 10
) -> np.ndarray:
    N = grid_size * grid_size
    agent_idx = np.ravel_multi_index((agent[1], agent[0]), (grid_size, grid_size))
    goal_idx = np.ravel_multi_index((goal[1], goal[0]), (grid_size, grid_size))
    oh_agent = np.zeros(N, dtype=np.float32)
    oh_agent[agent_idx] = 1.0
    oh_goal = np.zeros(N, dtype=np.float32)
    oh_goal[goal_idx] = 1.0
    return np.concatenate([oh_agent, oh_goal])


class OneHotFlatPosWrapper(ObservationWrapper):
    """Replace (agent,target) dict with a single concatenated one‑hot vector."""

    def __init__(self, env: gym.Env, grid_size: int = 10):
        super().__init__(env)
        self.grid_size = grid_size
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(grid_size * grid_size * 2,),
            dtype=np.float32,
        )

    def observation(self, obs):
        return onehot_agent_goal_positions(
            tuple(obs["agent"]), tuple(obs["goal"]), self.grid_size
        )


if __name__ == "__main__":
    print(onehot_agent_goal_positions((2, 0), (0, 0), 3))
