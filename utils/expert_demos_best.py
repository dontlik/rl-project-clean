"""
Batch expert demo generator for MiniGrid (e.g., DoorKey).

Features
- Rule-based expert: BFS navigation + orientation-aware action synthesis
- Phased plan: to KEY -> PICKUP -> near DOOR -> FACE door -> TOGGLE -> to GOAL
- Batch over seeds/episodes; save each episode to .npz
- Quality report: success rate, mean return, mean steps

Usage examples:
  python utils/generate_expert_demos.py --env_id MiniGrid-DoorKey-6x6-v0 \
      --seeds 0 1 2 --episodes_per_seed 10 --out_dir data/demos \
      --render 0

Optional tips:
- Set --goal_onehot to desired size if you use goal-conditioned inputs.
"""

import argparse, os, numpy as np, gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from collections import deque
from typing import List, Tuple, Optional

# MiniGrid action enum (stable across versions)
# left=0, right=1, forward=2, pickup=3, drop=4, toggle=5, done=6
A_LEFT, A_RIGHT, A_FWD, A_PICK, A_DROP, A_TOGGLE, A_DONE = 0, 1, 2, 3, 4, 5, 6

# ---------------------------------------
# Utilities: grid parsing & BFS over free cells
# ---------------------------------------
def get_free_mask(env) -> np.ndarray:
    """
    Return a 2D mask with 0 for free, 1 for blocked (walls, closed doors, lava, etc.)
    We'll treat DOOR as blocked when closed; the expert will explicitly stop next to it and toggle.
    """
    W, H = env.width, env.height
    mask = np.ones((W, H), dtype=np.int8)  # 1=blocked
    for i in range(W):
        for j in range(H):
            cell = env.grid.get(i, j)
            if cell is None:
                mask[i, j] = 0  # empty is free
            else:
                t = cell.type
                # walls/lava blocked; open doors are passable
                if t == 'wall' or t == 'lava':
                    mask[i, j] = 1
                elif t == 'door':
                    mask[i, j] = 0 if cell.is_open else 1
                else:
                    # objects (key, goal, etc.) occupy the cell but are passable for path planning
                    mask[i, j] = 0
    return mask

def bfs_free(mask: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """BFS shortest path on cells where mask==0. Returns list of (x,y) including start and goal. Empty if unreachable."""
    W, H = mask.shape
    if mask[start] == 1: return []
    Q = deque([start])
    parent = {start: None}
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    while Q:
        x,y = Q.popleft()
        if (x,y) == goal:
            break
        for dx,dy in moves:
            nx, ny = x+dx, y+dy
            if 0<=nx<W and 0<=ny<H and mask[nx,ny]==0 and (nx,ny) not in parent:
                parent[(nx,ny)] = (x,y)
                Q.append((nx,ny))
    if goal not in parent:
        return []
    path = []
    p = goal
    while p is not None:
        path.append(p)
        p = parent[p]
    return path[::-1]

# ---------------------------------------
# Orientation-aware action synthesis
# ---------------------------------------
DIR2VEC = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}  # MiniGrid: 0=right,1=down,2=left,3=up (may vary; infer from env.agent_dir vectors)

def vec_to_dir(dx, dy) -> int:
    # Map movement vector to direction id (right,down,left,up)
    if   (dx,dy)==(1,0):  return 0
    elif (dx,dy)==(0,1):  return 1
    elif (dx,dy)==(-1,0): return 2
    elif (dx,dy)==(0,-1): return 3
    else: raise ValueError("Invalid step vector")

def synth_move_actions(from_pos, from_dir, path: List[Tuple[int,int]]) -> List[int]:
    """
    Convert a path of positions into MiniGrid actions with turning and forward moves.
    - from_pos: (x,y) start
    - from_dir: int 0..3 current facing
    - path: positions including start and at least one step
    Returns: actions list; NOTE does NOT update env, only a local simulation of orientation.
    """
    if len(path) <= 1:
        return []
    actions = []
    cur_dir = from_dir
    cur = from_pos
    for nxt in path[1:]:
        dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
        target_dir = vec_to_dir(dx, dy)
        # rotate to face target_dir
        while cur_dir != target_dir:
            # choose left/right minimal rotation
            diff = (target_dir - cur_dir) % 4
            if diff == 1:  # turn right
                actions.append(A_RIGHT); cur_dir = (cur_dir+1)%4
            elif diff == 3:  # turn left
                actions.append(A_LEFT); cur_dir = (cur_dir-1)%4
            else:
                # either 180deg: two rights
                actions.append(A_RIGHT); cur_dir = (cur_dir+1)%4
        # move forward one cell
        actions.append(A_FWD)
        cur = nxt
    return actions, cur_dir, cur

# ---------------------------------------
# Object locators
# ---------------------------------------
def find_first(env, type_name: str) -> Optional[Tuple[int,int]]:
    W,H = env.width, env.height
    for i in range(W):
        for j in range(H):
            cell = env.grid.get(i,j)
            if cell is not None and cell.type == type_name:
                return (i,j)
    return None

def neighbors(p):
    x,y = p
    return [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

def closest_adjacent_free(mask, target: Tuple[int,int], start: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    """
    Find a free cell adjacent to target that is reachable from start.
    Useful for standing next to a door before toggling.
    """
    adj = [q for q in neighbors(target) if 0<=q[0]<mask.shape[0] and 0<=q[1]<mask.shape[1] and mask[q]==0]
    if not adj: return None
    # choose the one with shortest BFS distance
    best = None; best_len = 1e9
    for q in adj:
        path = bfs_free(mask, start, q)
        if path:
            if len(path) < best_len:
                best_len = len(path); best = q
    return best

# ---------------------------------------
# Phase planner for DoorKey-like tasks
# ---------------------------------------
def plan_episode_actions(env) -> List[int]:
    """
    High-level plan for DoorKey:
      1) path to KEY -> pickup
      2) path to cell adjacent to DOOR -> rotate to face door -> toggle
      3) path to GOAL
    Returns list of actions.
    """
    mask = get_free_mask(env)
    agent_pos = tuple(env.agent_pos)
    agent_dir = int(env.agent_dir)

    key_pos  = find_first(env, 'key')
    door_pos = find_first(env, 'door')
    goal_pos = find_first(env, 'goal')
    if key_pos is None or door_pos is None or goal_pos is None:
        raise RuntimeError("Missing key/door/goal in map")

    actions = []

    # 1) to KEY
    path1 = bfs_free(mask, agent_pos, key_pos)
    if not path1: raise RuntimeError("No path to key")
    a1, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path1)
    actions += a1
    # pickup
    actions += [A_PICK]

    # After pickup, door is still closed => still blocked in mask.
    # 2) go to an adjacent free cell next to door, face the door, toggle
    mask2 = get_free_mask(env)  # still closed door
    near = closest_adjacent_free(mask2, door_pos, agent_pos)
    if near is None: raise RuntimeError("No reachable adjacent cell near door")
    path2 = bfs_free(mask2, agent_pos, near)
    if not path2: raise RuntimeError("No path to door-adjacent")
    a2, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path2)
    actions += a2
    # face the door
    dx, dy = door_pos[0]-agent_pos[0], door_pos[1]-agent_pos[1]
    target_dir = vec_to_dir(np.sign(dx) if dx!=0 else 0, np.sign(dy) if dy!=0 else 0)
    while agent_dir != target_dir:
        diff = (target_dir - agent_dir) % 4
        if diff == 1: actions.append(A_RIGHT); agent_dir=(agent_dir+1)%4
        elif diff == 3: actions.append(A_LEFT); agent_dir=(agent_dir-1)%4
        else: actions.append(A_RIGHT); agent_dir=(agent_dir+1)%4
    # toggle to open
    actions += [A_TOGGLE]

    # 3) door now open; recompute free mask and go to GOAL
    mask3 = get_free_mask(env)  # if env updates only after step, the toggle will actually happen during execution;
                                # we add a small safety: if closed at runtime, step will open then path3 will still be valid as we move.
    path3 = bfs_free(mask3, agent_pos, goal_pos)
    if not path3:  # fallback: plan to door cell itself (in case mask didn't reflect open yet), then goal
        path_to_door = bfs_free(mask2, agent_pos, door_pos)
        if not path_to_door:
            raise RuntimeError("No path to door cell")
        a3a, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path_to_door)
        actions += a3a
        # then to goal
        mask_post = get_free_mask(env)
        path_post = bfs_free(mask_post, agent_pos, goal_pos)
        if not path_post: raise RuntimeError("No path to goal after door")
        a3b, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path_post)
        actions += a3b
    else:
        a3, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path3)
        actions += a3

    return actions

# ---------------------------------------
# Runner
# ---------------------------------------
def run_and_save(env_id: str, seeds: List[int], episodes_per_seed: int,
                 out_dir: str, render: bool=False, goal_onehot: int=0):
    os.makedirs(f"{out_dir}/{env_id}", exist_ok=True)
    report = []

    for seed in seeds:
        env_raw = gym.make(env_id)
        env = FullyObsWrapper(env_raw)
        for ep in range(episodes_per_seed):
            obs, info = env.reset(seed=seed*1000+ep)
            done = False
            total_r = 0.0
            steps = 0
            success = False

            # (optional) goal vector: fill with your encoding
            goal_vec = np.zeros((goal_onehot,), dtype=np.float32) if goal_onehot>0 else None

            # storage
            rec = {"obs":[], "goal":[], "action":[], "reward":[], "done":[], "info":[]}

            # plan actions once (for DoorKey-like; for stochastic maps you could replan during rollout)
            try:
                planned = plan_episode_actions(env.unwrapped)  # pass underlying minigrid env
            except Exception as e:
                # planning failed; skip episode
                print(f"[seed {seed} ep {ep}] planning failed: {e}")
                continue

            for a in planned:
                if render: env.render()
                # record obs before action
                obs_flat = obs["image"].astype(np.float32).flatten() / 10.0
                rec["obs"].append(obs_flat)
                if goal_vec is not None: rec["goal"].append(goal_vec.copy())
                rec["action"].append(int(a))

                obs, reward, terminated, truncated, info = env.step(a)
                total_r += float(reward)
                steps += 1
                done = bool(terminated or truncated)
                rec["reward"].append(float(reward))
                rec["done"].append(done)
                rec["info"].append(info)
                if done:
                    success = bool(info.get("success", False)) or bool(reward > 0)
                    break

            # finalize arrays & save
            for k in rec:
                if k == "info":
                    rec[k] = np.array(rec[k], dtype=object)
                elif k == "goal":
                    if goal_vec is None:
                        continue
                    rec[k] = np.array(rec[k], dtype=np.float32)
                else:
                    rec[k] = np.array(rec[k], dtype=np.float32 if k!="action" else np.int64)

            fname = f"{out_dir}/{env_id}/expert_seed{seed}_ep{ep}.npz"
            np.savez(fname, **rec, meta=np.array({"env":env_id, "seed":seed, "success":success}, dtype=object))
            report.append({"seed":seed, "ep":ep, "success":int(success), "return":total_r, "steps":steps})
            print(f"[seed {seed} ep {ep}] saved: {fname}  success={success}  return={total_r:.3f}  steps={steps}")

        env.close()

    # summary
    if len(report)==0:
        print("No episodes saved (all planning failed?)")
        return

    import pandas as pd
    df = pd.DataFrame(report)
    summary = {
        "episodes": len(df),
        "success_rate": float(df["success"].mean()),
        "mean_return": float(df["return"].mean()),
        "mean_steps": float(df["steps"].mean())
    }
    print("\n=== Expert Data Report ===")
    print(pd.DataFrame([summary]))
    df.to_csv(f"{out_dir}/{env_id}/expert_report.csv", index=False)
    print(f"Per-episode report saved to {out_dir}/{env_id}/expert_report.csv")

# ---------------------------------------
# CLI
# ---------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--episodes_per_seed", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="data/demos")
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--goal_onehot", type=int, default=0, help=">0 to store constant goal vector of this size")
    args = ap.parse_args()

    run_and_save(
        env_id=args.env_id,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        out_dir=args.out_dir,
        render=bool(args.render),
        goal_onehot=args.goal_onehot
    )
