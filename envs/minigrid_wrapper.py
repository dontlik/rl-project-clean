import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import numpy as np

def make_env(env_id="MiniGrid-DoorKey-6x6-v0", seed=0, goal_spec=None):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = FullyObsWrapper(env)   # get full map observation
    env.goal_spec = goal_spec
    return env

def obs_to_tensor(obs_dict):
    obs = obs_dict["image"].flatten() / 10.0
    goal = obs_dict.get("goal", np.zeros(6))
    return np.concatenate([obs, goal])
