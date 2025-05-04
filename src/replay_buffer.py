import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int):
        self.max_size = max_size

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int32)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)

        b_states = self.states[idxs]
        b_actions = self.actions[idxs]
        b_rewards = self.rewards[idxs]
        b_next_states = self.next_states[idxs]
        b_dones = self.dones[idxs]

        return (
            jnp.array(b_states),
            jnp.array(b_actions),
            jnp.array(b_rewards),
            jnp.array(b_next_states),
            jnp.array(b_dones),
        )

    def __len__(self) -> int:
        return self.size
