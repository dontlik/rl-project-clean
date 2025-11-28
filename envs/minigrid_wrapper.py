import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import numpy as np
import torch
import torch.nn.functional as F
def make_env(env_id="MiniGrid-DoorKey-6x6-v0", seed=0, goal_spec=None):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = FullyObsWrapper(env)   # get full map observation
    env.goal_spec = goal_spec
    return env

def resize(obs_img, target_shape=(3, 15, 15)):

    img = torch.tensor(obs_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 10.0
    resized = F.interpolate(img, size=target_shape[1:], mode='bilinear', align_corners=False)
    return resized.squeeze(0) 

def obs_to_tensor(obs_dict,goal_dim):
    obs = resize(obs_dict["image"])
    if goal_dim > 0:
        goal_vec = obs_dict.get("goal", np.zeros(goal_dim, dtype=np.float32))
        goal = torch.tensor(goal_vec, dtype=torch.float32, device=device)  # [1, goal_dim]
    else:
        goal = None
    return obs, goal

