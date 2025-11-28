import argparse, os, numpy as np, gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from collections import deque
from typing import List, Tuple, Optional
import minigrid
import torch
import torch.nn.functional as F

A_LEFT, A_RIGHT, A_FWD, A_PICK, A_DROP, A_TOGGLE, A_DONE = 0, 1, 2, 3, 4, 5, 6

def get_free_mask(env) -> np.ndarray:
    W, H = env.width, env.height
    mask = np.ones((W, H), dtype=np.int8)  # 1=blocked
    for i in range(W):
        for j in range(H):
            cell = env.grid.get(i, j)
            if cell is None:
                mask[i, j] = 0
            else:
                t = cell.type
                if t in ("wall", "lava"):
                    mask[i, j] = 1
                elif t == "door":
                    mask[i, j] = 0 if cell.is_open else 1
                else:
                    mask[i, j] = 0
    return mask

def bfs_free(mask: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """Shortest path on cells where mask==0. Returns [] if unreachable."""
    W, H = mask.shape
    if not (0 <= start[0] < W and 0 <= start[1] < H and 0 <= goal[0] < W and 0 <= goal[1] < H):
        return []
    if mask[start] == 1 or mask[goal] == 1:
        return []
    Q = deque([start])
    parent = {start: None}
    moves = [(1,0),(-1,0),(0,1),(0,-1)]
    # ✅ FIX: use popleft to avoid mutating during iteration
    while Q:
        x, y = Q.popleft()
        if (x, y) == goal:
            break
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and mask[nx, ny] == 0 and (nx, ny) not in parent:
                parent[(nx, ny)] = (x, y)
                Q.append((nx, ny))
    if goal not in parent:
        return []
    path = []
    p = goal
    while p is not None:
        path.append(p)
        p = parent[p]
    return path[::-1]

def vec_to_dir(dx:int, dy:int) -> int:
    key = (int(np.sign(dx)), int(np.sign(dy)))
    if key==(1,0): return 0
    if key==(0,1): return 1
    if key==(-1,0): return 2
    if key==(0,-1): return 3
    raise ValueError(f"Invalid step vector {dx,dy}")

def synth_move_actions(from_pos: Tuple[int,int],
                       from_dir: int,
                       path: List[Tuple[int,int]]
                       ) -> Tuple[List[int], int, Tuple[int,int]]:
    if len(path) <= 1:
        return [], from_dir, from_pos
    actions: List[int] = []
    cur_dir = from_dir
    cur = from_pos
    for nxt in path[1:]:
        dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
        tgt = vec_to_dir(dx, dy)
        while cur_dir != tgt:
            diff = (tgt - cur_dir) % 4
            if diff == 1: actions.append(A_RIGHT); cur_dir=(cur_dir+1)%4
            elif diff == 3: actions.append(A_LEFT);  cur_dir=(cur_dir-1)%4
            else:           actions.append(A_RIGHT); cur_dir=(cur_dir+1)%4
        actions.append(A_FWD)
        cur = nxt
    return actions, cur_dir, cur

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
    adj = [q for q in neighbors(target)
           if 0<=q[0]<mask.shape[0] and 0<=q[1]<mask.shape[1] and mask[q]==0]
    if not adj: return None
    best=None; best_len=1e9
    for q in adj:
        path=bfs_free(mask,start,q)
        if path and len(path)<best_len:
            best_len=len(path); best=q
    return best

def plan_episode_actions(env) -> List[int]:
    """
    State-aware planner:
      - If no key in hand: go to KEY and PICKUP
      - If door is closed: go to door-adjacent (or door cell), FACE door, TOGGLE
      - Then: go to GOAL
    Plan from *current* env state each call (used with iterative re-planning).
    """
    actions: List[int] = []

    mask = get_free_mask(env)
    agent_pos = tuple(env.agent_pos)
    agent_dir = int(env.agent_dir)
    key_pos  = find_first(env, 'key')
    door_pos = find_first(env, 'door')
    goal_pos = find_first(env, 'goal')
    if door_pos is None or goal_pos is None:
        raise RuntimeError("Missing door/goal in map")

    # Phase 1: get key
    has_key = (env.carrying is not None) and (getattr(env.carrying, "type", "") == "key")
    if (not has_key) and (key_pos is not None):
        path1 = bfs_free(mask, agent_pos, key_pos)
        if not path1: raise RuntimeError("No path to key")
        a1, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path1)
        actions += a1
        actions.append(A_PICK)
        return actions

    # Phase 2: open door if closed
    door_cell = env.grid.get(door_pos[0], door_pos[1])
    is_open = (door_cell is not None) and getattr(door_cell, "is_open", False)
    if not is_open:
        mask2 = get_free_mask(env)  # door closed => blocked
        near = closest_adjacent_free(mask2, door_pos, agent_pos)
        if near is not None:
            path2 = bfs_free(mask2, agent_pos, near)
            if not path2: raise RuntimeError("No path to door-adjacent")
            a2, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path2)
            actions += a2
        else:
            mask2_allow = mask2.copy(); mask2_allow[door_pos] = 0
            path2 = bfs_free(mask2_allow, agent_pos, door_pos)
            if not path2: raise RuntimeError("No path to door cell")
            a2, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path2)
            actions += a2
        # face the door
        dx, dy = door_pos[0]-agent_pos[0], door_pos[1]-agent_pos[1]
        if abs(dx)+abs(dy) > 0:
            tgt = vec_to_dir(dx if dx!=0 else 0, dy if dy!=0 else 0)
            while agent_dir != tgt:
                diff=(tgt-agent_dir)%4
                if diff==1: actions.append(A_RIGHT); agent_dir=(agent_dir+1)%4
                elif diff==3: actions.append(A_LEFT);  agent_dir=(agent_dir-1)%4
                else:         actions.append(A_RIGHT); agent_dir=(agent_dir+1)%4
        actions.append(A_TOGGLE)
        return actions

    # Phase 3: go to goal (assume door open)
    mask_open = get_free_mask(env)
    mask_open[door_pos] = 0  # ensure door cell is treated as free
    path3 = bfs_free(mask_open, agent_pos, goal_pos)
    if not path3: raise RuntimeError("No path to goal")
    a3, agent_dir, agent_pos = synth_move_actions(agent_pos, agent_dir, path3)
    actions += a3
    return actions

def resize(obs_img, target_shape=(3, 15, 15)):

    img = torch.tensor(obs_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 10.0
    resized = F.interpolate(img, size=target_shape[1:], mode='bilinear', align_corners=False)
    return resized.squeeze(0) 


def run_and_save(env_id: str, seeds: List[int], episodes_per_seed: int,
                 out_dir: str, render: bool=False, goal_onehot: int=0):
    os.makedirs(f"{out_dir}/{env_id}", exist_ok=True)
    report = []

    for seed in seeds:
        env_raw = gym.make(env_id)
        env = FullyObsWrapper(env_raw)
        for ep in range(episodes_per_seed):
            obs, info = env.reset(seed=seed*1000+ep)

            grid = env.unwrapped.grid

            total_r = 0.0
            steps = 0
            success = False
            goal_vec = np.zeros((goal_onehot,), dtype=np.float32) if goal_onehot>0 else None
            rec = {"obs":[], "goal":[], "action":[], "reward":[], "done":[], "info":[]}

            try:
                done = False
                loops = 0
                max_loops = 8  # avoid infinite loops
                while not done and loops < max_loops:
                    loops += 1
                    planned = plan_episode_actions(env.unwrapped)
                    for a in planned:
                        if render: env.render()
                        # record obs BEFORE action
                        #obs_flat = obs["image"].astype(np.float32).flatten() / 10.0
                        #rec["obs"].append(obs_flat)
                        
                        
                        obs_resized = resize(obs["image"])
                        rec["obs"].append(obs_resized)

                        resized = resize(obs["image"])
                        print(resized.shape)

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
                            success = bool(info.get("success", False)) or (total_r > 0)
                            break
                    # loop to re-plan from the *current* state
            except Exception as e:
                print(f"[seed {seed} ep {ep}] planning failed: {e}")
                continue

            # finalize & save
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--episodes_per_seed", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="data/demos")
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--goal_onehot", type=int, default=0)
    args = ap.parse_args()
    

    
    
    run_and_save(
        env_id=args.env_id,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        out_dir=args.out_dir,
        render=bool(args.render),
        goal_onehot=args.goal_onehot
    )
