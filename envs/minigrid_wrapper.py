import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper

def make_env(env_id="MiniGrid-DoorKey-6x6-v0", seed=0, goal_spec=None):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = FullyObsWrapper(env)  # or RGBImgPartialObsWrapper(env)
    if goal_spec is not None:
        env.goal_spec = goal_spec  # store for future use
    return env

def obs_to_tensor(obs_dict):
    # flatten observation into vector
    obs = obs_dict["image"].flatten() / 10.0  # normalization
    if "goal" in obs_dict:
        obs = np.concatenate([obs, obs_dict["goal"]])
    return obs
