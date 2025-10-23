import argparse, os, numpy as np, gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from collections import deque

A_LEFT, A_RIGHT, A_FWD, A_PICK, A_DROP, A_TOGGLE, A_DONE = 0,1,2,3,4,5,6
DIR2VEC = {0:(1,0),1:(0,1),2:(-1,0),3:(0,-1)}

def vec_to_dir(dx,dy):
    return {(1,0):0,(0,1):1,(-1,0):2,(0,-1):3}[(np.sign(dx),np.sign(dy))]

def get_free_mask(env):
    W,H = env.width, env.height
    m = np.ones((W,H),dtype=np.int8)
    for i in range(W):
        for j in range(H):
            cell = env.grid.get(i,j)
            if cell is None: m[i,j]=0
            else:
                if cell.type in ("wall","lava"): m[i,j]=1
                elif cell.type=="door": m[i,j]=0 if cell.is_open else 1
                else: m[i,j]=0
    return m

def bfs(mask,start,goal):
    if mask[start]==1: return []
    W,H = mask.shape
    Q=deque([start]); parent={start:None}
    for (x,y) in Q:
        if (x,y)==goal: break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<W and 0<=ny<H and mask[nx,ny]==0 and (nx,ny) not in parent:
                parent[(nx,ny)]=(x,y); Q.append((nx,ny))
    if goal not in parent: return []
    path=[]; p=goal
    while p is not None: path.append(p); p=parent[p]
    return path[::-1]

def synth_moves(pos, direc, path):
    acts=[]
    cur_dir=direc; cur=pos
    for nxt in path[1:]:
        dx,dy=nxt[0]-cur[0],nxt[1]-cur[1]
        tgt=vec_to_dir(dx,dy)
        while cur_dir!=tgt:
            diff=(tgt-cur_dir)%4
            if diff==1: acts.append(A_RIGHT); cur_dir=(cur_dir+1)%4
            elif diff==3: acts.append(A_LEFT); cur_dir=(cur_dir-1)%4
            else: acts.append(A_RIGHT); cur_dir=(cur_dir+1)%4
        acts.append(A_FWD); cur=nxt
    return acts, cur_dir, cur

def find_first(env, t):
    for i in range(env.width):
        for j in range(env.height):
            cell = env.grid.get(i,j)
            if cell is not None and cell.type==t: return (i,j)
    return None

def closest_adjacent_free(mask,target,start):
    cands=[(target[0]+dx,target[1]+dy) for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]
           if 0<=target[0]+dx<mask.shape[0] and 0<=target[1]+dy<mask.shape[1]
           and mask[target[0]+dx,target[1]+dy]==0]
    best=None; L=1e9
    for q in cands:
        p=bfs(mask,start,q)
        if p and len(p)<L: best=q; L=len(p)
    return best

def plan(env):
    mask=get_free_mask(env.unwrapped)
    apos=tuple(env.unwrapped.agent_pos); adir=int(env.unwrapped.agent_dir)
    key=find_first(env.unwrapped,'key'); door=find_first(env.unwrapped,'door'); goal=find_first(env.unwrapped,'goal')
    assert key and door and goal, "objects not found"
    acts=[]

    p1=bfs(mask,apos,key); a1,adir,apos=synth_moves(apos,adir,p1); acts+=a1; acts.append(A_PICK)
    mask2=get_free_mask(env.unwrapped)
    near=closest_adjacent_free(mask2,door,apos); p2=bfs(mask2,apos,near)
    a2,adir,apos=synth_moves(apos,adir,p2); acts+=a2
    # face door
    dx,dy=door[0]-apos[0],door[1]-apos[1]; tgt=vec_to_dir(dx,dy)
    while adir!=tgt:
        diff=(tgt-adir)%4
        if diff==1: acts.append(A_RIGHT); adir=(adir+1)%4
        elif diff==3: acts.append(A_LEFT); adir=(adir-1)%4
        else: acts.append(A_RIGHT); adir=(adir+1)%4
    acts.append(A_TOGGLE)
    mask3=get_free_mask(env.unwrapped)
    p3=bfs(mask3,apos,goal)
    if not p3: p3=bfs(mask2,apos,door)
    a3,_,_=synth_moves(apos,adir,p3); acts+=a3
    return acts

def run(env_id, seeds, eps_per_seed, out_dir, goal_dim=0):
    os.makedirs(f"{out_dir}/{env_id}", exist_ok=True)
    report=[]
    for sd in seeds:
        env = FullyObsWrapper(gym.make(env_id))
        for ep in range(eps_per_seed):
            obs, info = env.reset(seed=sd*1000+ep)
            traj={"obs":[], "action":[], "reward":[], "done":[], "info":[]}
            if goal_dim>0:
                traj["goal"]=[]
                goal_vec=np.zeros(goal_dim, dtype=np.float32)
            try:
                plan_actions=plan(env)
            except Exception as e:
                print(f"[seed {sd} ep {ep}] plan failed: {e}"); continue
            total=0; steps=0; success=False
            for a in plan_actions:
                obs_flat=obs["image"].astype(np.float32).flatten()/10.0
                traj["obs"].append(obs_flat)
                if goal_dim>0: traj["goal"].append(goal_vec.copy())
                traj["action"].append(int(a))
                obs, r, term, trunc, info = env.step(a)
                total+=float(r); steps+=1; done=bool(term or trunc)
                traj["reward"].append(float(r)); traj["done"].append(done); traj["info"].append(info)
                if done: success = bool(info.get("success", False)) or (r>0); break
            for k in traj:
                if k=="info": traj[k]=np.array(traj[k], dtype=object)
                elif k=="action": traj[k]=np.array(traj[k], dtype=np.int64)
                else: traj[k]=np.array(traj[k], dtype=np.float32)
            path=f"{out_dir}/{env_id}/expert_seed{sd}_ep{ep}.npz"
            np.savez(path, **traj, meta=np.array({"env":env_id,"seed":sd,"success":success},dtype=object))
            report.append((sd,ep,success,total,steps))
            print(f"[seed {sd} ep {ep}] saved: {path}  success={success}  return={total:.2f}  steps={steps}")
        env.close()
    # write a small CSV
    import pandas as pd
    df=pd.DataFrame(report, columns=["seed","ep","success","return","steps"])
    df.to_csv(f"{out_dir}/{env_id}/expert_report.csv", index=False)
    print(df.groupby("seed")[["success","return","steps"]].mean())

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--episodes_per_seed", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="data/demos")
    ap.add_argument("--goal_dim", type=int, default=0)
    args=ap.parse_args()
    run(args.env_id, args.seeds, args.episodes_per_seed, args.out_dir, goal_dim=args.goal_dim)
