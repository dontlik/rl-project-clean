import gymnasium as gym
import pygame
import numpy as np
import os
import minigrid
import argparse


KEY_TO_ACTION = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 1,
    pygame.K_UP: 2,
    pygame.K_DOWN: 3,
    pygame.K_TAB: 4,
    pygame.K_SPACE: 5,
    pygame.K_RETURN: 6,
}

def run(env_id,save_dir):
    os.makedirs(save_dir, exist_ok=True)


    obs, act, reward, goal = [], [], [], []

    pygame.init()
    env = gym.make(env_id, render_mode="human")

    demo_idx = 0
    collecting = True
    goal_dim = 0 
    while collecting:
        obs, _ = env.reset()
        obs_list, act_list, reward_list, goal_list = [], [], [], []
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    collecting = False
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        collecting = False
                        done = True
                        break
                    elif event.key in KEY_TO_ACTION:
                        action = KEY_TO_ACTION[event.key]

                        obs_img = obs["image"].transpose(2, 0, 1)
                        obs_list.append(obs_img)
                        act_list.append(action)
                        goal_list.append(np.zeros(goal_dim, dtype=np.float32))

                        obs, reward, terminated, truncated, _ = env.step(action)
                        print("Action:", action, "| Reward:", reward)
                        reward_list.append(reward)

                        if terminated or truncated:
                            done = True
                            break
        if len(act_list) > 0:
            save_path = os.path.join(save_dir, f"demo_{demo_idx:03d}.npz")
            np.savez(
                save_path,
                obs=np.array(obs_list, dtype=np.uint8),
                action=np.array(act_list, dtype=np.int64),
                reward=np.array(reward_list, dtype=np.float32),
                goal=np.array(goal_list, dtype=np.float32),
            )
            print(f" Saved {save_path}")
            demo_idx += 1
            
    env.close()

if __name__=="__main__":

    ap=argparse.ArgumentParser()
    ap.add_argument("--env_id",type=str,default="MiniGrid-MultiRoom-N2-S4-v0")
    ap.add_argument("--save_dir",type=str,default="data/demos/manual")
    args=ap.parse_args()
    run(args.env_id, args.save_dir)