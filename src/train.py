import os
from typing import Sequence, Union

import gymnasium as gym
from gym.wrappers import RecordVideo
from loguru import logger
from tqdm.notebook import tqdm

import wandb
from src.agent import DeepQLearningAgent
from src.config import RLConfig


def evaluate_agent(env: gym.Env, agent: DeepQLearningAgent) -> float:
    old_eps = agent.eps
    agent.eps = 0.0  # Set greedy policy

    state, _ = env.reset(seed=33)
    total_reward = 0.0
    done = False

    while not done:
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    agent.eps = old_eps  # Restore exploration parameter
    return total_reward


def train_dql_agent(
    config: RLConfig,
    env: gym.Env,
    state_dim: Union[Sequence[int] | int],
    eval_env: gym.Env | None = None,
    video_dir: str = "./videos",
):
    logger.info("Starting training with config: {}", config)

    action_dim = env.action_space.n
    logger.info("State dim: {}, Action dim: {}", state_dim, action_dim)

    agent = DeepQLearningAgent(
        config=config,
        state_dim=state_dim,
        action_dim=action_dim,
        decay_steps=config.decay_steps,
    )

    with wandb.init(project="rl-playground", config=config.model_dump()) as _:
        wandb.define_metric("train/episode_reward", step_metric="episode")
        wandb.define_metric("epsilon", step_metric="global_step")
        wandb.define_metric("eval/total_reward", step_metric="episode")
        wandb.define_metric("eval/video", step_metric="episode")

        for ep in tqdm(range(1, config.n_episodes + 1), desc="Episodes"):
            state, _ = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                _ = agent.learn(state, action, reward, next_state, done)
                ep_reward += reward
                state = next_state

                wandb.log({"global_step": agent.step, "epsilon": agent.eps})

            wandb.log(
                {
                    "episode": ep,
                    "train/episode_reward": ep_reward,
                }
            )

            if ep % config.evaluation_frequency == 0:
                video_env = RecordVideo(
                    eval_env if eval_env else env,
                    video_dir,
                    episode_trigger=lambda e: True,
                )
                eval_reward = evaluate_agent(video_env, agent)
                video_env.close()

                video_path = os.path.join(video_dir, "rl-video-episode-0.mp4")
                wandb.log(
                    {
                        "eval/total_reward": eval_reward,
                        "eval/video": wandb.Video(video_path, format="gif"),
                    }
                )

    env.close()
