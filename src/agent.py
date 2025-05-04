from typing import Sequence, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.config import RLConfig

from .net import ConvNetwork
from .replay_buffer import ReplayBuffer


@nnx.jit
def train_step(
    optim: nnx.Optimizer,
    target_model: nnx.Module,
    states: jnp.ndarray,
    actions: int,
    rewards: float,
    next_states: jnp.ndarray,
    dones: bool,
    gamma: float,
):
    def loss_fn(model: nnx.Module):
        q_s = model(states)  # Q(s,·)
        q_sa = jnp.take_along_axis(q_s, actions[:, None], axis=1).squeeze()  # Q(s,a)
        next_q_s = target_model(next_states)  # Q(s',·)
        max_next = jax.lax.stop_gradient(jnp.max(next_q_s, axis=1))  # max_a' Q(s',a')
        target = rewards + gamma * max_next * (1.0 - dones)
        return jnp.mean((q_sa - target) ** 2)  # MSE

    loss, grads = nnx.value_and_grad(loss_fn)(optim.model)
    optim.update(grads)
    return loss


class DeepQLearningAgent:
    def __init__(
        self,
        config: RLConfig,
        state_dim: Union[Sequence[int], int],
        action_dim: int,
        decay_steps: int,
    ):
        self.rng = np.random.default_rng(config.seed or 0)
        self.action_dim = action_dim
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.decay_steps = decay_steps
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.warmup_start_size = config.warmup_start_size
        self.target_update_frequency = config.target_update_frequency
        self.step = 0

        q_net = ConvNetwork(state_dim, action_dim, rngs=nnx.Rngs(config.seed))
        # q_net = GridNet(state_dim, action_dim, rngs=nnx.Rngs(config.seed))
        # q_net = FullyConnectedNetwork(
        #     state_dim, config.hidden_dim, action_dim, rngs=nnx.Rngs(config.seed)
        # )
        tx = optax.adam(config.lr)
        self.optim = nnx.Optimizer(q_net, tx)
        self.target_q_net = ConvNetwork(
            state_dim, action_dim, rngs=nnx.Rngs(config.seed)
        )
        # self.target_q_net = GridNet(state_dim, action_dim, rngs=nnx.Rngs(config.seed))
        # self.target_q_net = FullyConnectedNetwork(
        #     state_dim, config.hidden_dim, action_dim, rngs=nnx.Rngs(config.seed)
        # )
        self.update_target()
        self.replay_buffer = ReplayBuffer(config.replay_memory_size, state_dim)

    def update_target(self):
        """Copy parameters from online network to target network."""
        _, params = nnx.split(self.optim.model, nnx.Param)
        params_copy = jax.tree.map(lambda x: x.copy(), params)
        nnx.update(self.target_q_net, params_copy)

    def linear_schedule(
        self, eps_start: float, eps_end: float, decay_steps: int, step: int
    ):
        eps_delta = (eps_start - eps_end) / decay_steps
        return max(eps_end, eps_start - eps_delta * step)

    def act(self, state: np.ndarray) -> int:
        self.eps = self.linear_schedule(
            self.eps_start, self.eps_end, self.decay_steps, self.step
        )
        # Exploration
        if self.rng.random() < self.eps:
            return self.rng.integers(0, self.action_dim)
        # Exploitation
        x = jnp.asarray(state)
        if len(x.shape) == 3:
            x = jnp.expand_dims(x, axis=0)
        q_s = self.optim.model(x)
        return int(jnp.argmax(q_s[0]))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.warmup_start_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        loss = train_step(
            self.optim,
            self.target_q_net,
            states,
            actions,
            rewards,
            next_states,
            dones,
            self.gamma,
        )

        self.step += 1

        if self.step % self.target_update_frequency == 0:
            self.update_target()

        return float(loss)
