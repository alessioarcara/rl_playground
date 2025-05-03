import wandb
from wandb.sdk.wandb_config import Config


def init_wandb() -> Config:
    wandb.init(
        project="rl-playground",
        config={
            "batch_size": 32,
            "lr": 1e-3,
            "gamma": 0.99,
            "eps_start": 1.0,
            "eps_end": 0.05,
            "hidden_dim": 128,
            "n_episodes": 500,
            "max_steps_per_episode": 32,
            "grid_size": 5,
            "seed": 0,
            "target_update_frequency": 4,
            "evaluation_frequency": 50,
        },
    )
    return wandb.config
