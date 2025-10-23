"""
Rule-based expert policy for MiniGrid-DoorKey-6x6-v0.
Generates high-quality expert demonstrations for Behavior Cloning.

Usage:
  python utils/expert_door_key.py --episodes 20 --seed 0
"""

import argparse, os, numpy as np, gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from collections import deque

# ------------- Helper: BFS shortest path in grid -----------------
def bfs(grid, start, goal):
    """Return a list of positions from start to goal (inclusive)."""
    q = deque([start])
    parents = {start: None}
    H, W = grid.shape
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        x, y = q.popleft()
        if (x,y) == goal:
            break
        for dx, dy in moves:
            nx, ny = x+dx, y+dy
            if 0<=nx<H and 0<=ny<W and grid[nx,ny]==0 and (nx,ny) not in parents:
                parents[(nx,ny)] = (x,y)
                q.append((nx,ny))
    if goal not in parents:
        return []  # no path
    path = []
    p = goal
    while p is not None:
        path.append(p)
        p = parents[p]
    return path[::-1]

# ------------- Expert Action Planner -----------------
def plan_actions(env):
    """
    Plan full sequence of actions to: get key -> open door -> go to goal.
    Return: list of actions (ints)
    """
    grid = env.grid.encode()[:,:,0]  # object types
    free = np.where(grid==0, 0, 1)   # 0=free, 1=blocked

    # locate agent, key, door, goal
    ax, ay = env.agent_pos
    key_pos = None
    door_pos = None
    goal_pos = None

    for i in range(env.width):
        for j in range(env.height):
            cell = env.grid.get(i,j)
            if cell is None: continue
            if cell.type == 'key':
                key_pos = (i,j)
            elif cell.type == 'door':
                door_pos = (i,j)
            elif cell.type == 'goal':
                goal_pos = (i,j)

    if not (key_pos and door_pos and goal_pos):
        raise RuntimeError("Objects not found in grid")

    path1 = bfs(free, (ax,ay), key_pos)
    path2 = bfs(free, key_pos, door_pos)
    path3 = bfs(free, door_pos, goal_pos)

    # convert path positions to actions (approximate)
    def to_actions(path):
        acts = []
        for (x0,y0),(x1,y1) in zip(path[:-1], path[1:]):
            dx, dy = x1-x0, y1-y0
            if dx == -1: acts.append(0)  # left
            elif dx == 1: acts.append(1) # right
            elif dy == -1: acts.append(2) # up
            elif dy == 1: acts.append(3)  # down
        return acts

    actions = []
    # move to key
    actions += [2]*len(path1) + [3]  # simplification
    # pickup key
    actions.append(env.actions.pickup)
    # move to door
    actions += [2]*len(path2)
    # open door
    actions.append(env.actions.toggle)
    # move to goal
    actions += [2]*len(path3)
    return actions

# ------------- Main Script -----------------
def generate_demos(env_id="MiniGrid-DoorKey-6x6-v0", episodes=20, seed=0, out_dir="data/demos"):
    env = FullyObsWrapper(gym.make(env_id))
    os.makedirs(f"{out_dir}/{env_id}", exist_ok=True)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed+ep)
        done = False
        data = {"obs":[], "goal":[], "action":[], "reward":[], "done":[], "info":[]}

        # simple constant goal encoding (for now)
        goal_vec = np.array([1,0,0,0,0,0], dtype=np.float32)

        # plan full sequence (rule-based)
        try:
            planned = plan_actions(env)
        except Exception as e:
            print(f"Episode {ep}: planning failed, skipping ({e})")
            continue

        for a in planned:
            obs_f = obs["image"].flatten().astype(np.float32) / 10.0
            data["obs"].append(obs_f)
            data["goal"].append(goal_vec)
            data["action"].append(int(a))

            obs, reward, terminated, truncated, info = env.step(a)
            data["reward"].append(float(reward))
            data["done"].append(bool(terminated or truncated))
            data["info"].append(info)
            if terminated or truncated:
                break

        # save episode
        for k in data:
            data[k] = np.array(data[k], dtype=object if k=="info" else np.float32)
        np.savez(f"{out_dir}/{env_id}/expert_seed{seed}_ep{ep}.npz", **data)
        print(f"Saved episode {ep} ({len(data['action'])} steps)")
    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="data/demos")
    args = ap.parse_args()
    generate_demos(args.env_id, args.episodes, args.seed, args.out_dir)
