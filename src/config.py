from pathlib import Path

import yaml
from pydantic import BaseModel


class RLConfig(BaseModel):
    batch_size: int
    lr: float
    gamma: float
    eps_start: float
    eps_end: float
    decay_steps: int
    hidden_dim: int
    n_episodes: int
    max_steps_per_episode: int
    grid_size: int | None
    seed: int
    target_update_frequency: int
    evaluation_frequency: int
    replay_memory_size: int
    warmup_start_size: int


def _load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).expanduser().read_text()) or {}


def build_cfg(*yaml_paths: str | Path) -> RLConfig:
    merged = {}
    for p in yaml_paths:  # ordine = priorità (l’ultimo vince)
        merged.update(_load_yaml(p))
    return RLConfig(**merged)  # validazione
