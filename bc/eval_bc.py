import argparse, numpy as np, torch as th, pandas as pd
from envs.minigrid_wrapper import make_env #obs_to_vec
from .train_bc import Policy

def load_policy(ckpt):
    d=th.load(ckpt, map_location="cpu")
    pi=Policy(d["in_dim"], 7)  # DoorKey is 7 actions
    pi.load_state_dict(d["state_dict"]); pi.eval()
    return pi, d["in_dim"]

def run(env_id, seeds, episodes, ckpt):
    pi,in_dim=load_policy(ckpt)
    device=th.device("cuda" if th.cuda.is_available() else "cpu")
    pi.to(device)
    stats=[]
    for sd in seeds:
        env=make_env(env_id, seed=sd)
        for _ in range(episodes):
            obs, info = env.reset(seed=sd)
            ret=0; done=False; steps=0
            while not done:
                x=obs["image"].astype(np.float32).flatten()/10.0
                xt=th.tensor(x, dtype=th.float32, device=device).unsqueeze(0)
                with th.no_grad():
                    a=pi(xt).argmax(-1).item()
                obs, r, term, trunc, info = env.step(a)
                ret+=float(r); steps+=1; done=bool(term or trunc)
            stats.append({"seed":sd, "return":ret, "success": int(info.get("success", False) or ret>0), "steps":steps})
    df=pd.DataFrame(stats); print(df.describe())
    print("success rate:", df["success"].mean())
    return df

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--env_id",type=str,default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--episodes",type=int,default=50)
    ap.add_argument("--ckpt",type=str,default="bc/checkpoints/pi_ref.pt")
    args=ap.parse_args()
    run(args.env_id, args.seeds, args.episodes, args.ckpt)

