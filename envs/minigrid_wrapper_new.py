import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import numpy as np

def make_env(env_id="MiniGrid-DoorKey-6x6-v0", seed=0, goal_spec=None, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)  # render_mode="human" to watch
    env.reset(seed=seed)
    env = FullyObsWrapper(env)  # full 2D grid tensor in obs["image"]
    env.goal_spec = goal_spec  # optional: store goal encoding
    return env

def obs_to_vec(obs_dict, goal_dim=0):
    img = obs_dict["image"].astype(np.float32).flatten() / 10.0  # simple scaling
    if goal_dim > 0:
        goal = obs_dict.get("goal", np.zeros(goal_dim, dtype=np.float32))
        return np.concatenate([img, goal], axis=0)
    return img

def sample_valid_action(env):
    # for safety fallback; MiniGrid actions are 0..6
    return env.action_space.sample()
