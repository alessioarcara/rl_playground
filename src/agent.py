import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .net import QNetwork


@nnx.jit
def train_step(
    optim: nnx.Optimizer,
    target_model: QNetwork,
    state: jnp.ndarray,
    action: int,
    reward: float,
    next_state: jnp.ndarray,
    done: bool,
    gamma: float,
):
    def loss_fn(model: QNetwork):
        q_s = model(state)  # Q(s,·)
        q_sa = q_s[action]  # Q(s,a)
        next_q_s = target_model(next_state)  # Q(s',·)
        max_next = jax.lax.stop_gradient(jnp.max(next_q_s))  # max_a' Q(s',a')
        target = reward + gamma * max_next * (1.0 - done)
        return jnp.mean((q_sa - target) ** 2)  # MSE

    loss, grads = nnx.value_and_grad(loss_fn)(optim.model)
    optim.update(grads)
    return loss


class DeepQLearningAgent:
    def __init__(
        self,
        config,
        state_dim: int,
        action_dim: int,
        decay_steps: int,
    ):
        self.rng = np.random.default_rng(config.seed or 0)
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.decay_steps = decay_steps
        self.action_dim = action_dim
        self.gamma = config.gamma
        self.target_update_frequency = config.target_update_frequency

        q_net = QNetwork(
            state_dim, config.hidden_dim, action_dim, rngs=nnx.Rngs(config.seed)
        )
        tx = optax.adam(config.lr)
        self.optim = nnx.Optimizer(q_net, tx)
        self.target_q_net = QNetwork(
            state_dim, config.hidden_dim, action_dim, rngs=nnx.Rngs(config.seed)
        )
        self.update_target()
        self.step = 0

    def update_target(self):
        """Copia i parametri dalla rete online a quella target."""
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
            action = self.rng.integers(0, self.action_dim)
        # Exploitation
        else:
            q_s = self.optim.model(jnp.asarray(state))
            action = int(jnp.argmax(q_s))

        self.step += 1
        return action

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        loss = train_step(
            self.optim,
            self.target_q_net,
            jnp.asarray(state),
            action,
            reward,
            jnp.asarray(next_state),
            done,
            self.gamma,
        )

        if self.step % self.target_update_frequency == 0:
            self.update_target()

        return float(loss)
