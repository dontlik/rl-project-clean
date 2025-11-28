import argparse, numpy as np, torch as th, pandas as pd
from envs.minigrid_wrapper import make_env, obs_to_tensor #obs_to_vec
from .train_bc import Policy, Cnn_Emb
import torch
def load_policy(ckpt):
    d=th.load(ckpt, map_location="cpu")
    in_dim = d["in_dim"]
    goal_dim = d["goal_dim"]
    cnn_out_dim = in_dim - goal_dim
    encoder = Cnn_Emb(in_dim=(3, 15, 15), out_dim=cnn_out_dim)

    pi=Policy(n_actions=7, in_dim=d["in_dim"])  # DoorKey is 7 actions
    encoder.load_state_dict(d["encoder"])
    pi.load_state_dict(d["state_dict"]); 
    encoder.eval()
    pi.eval()
    return encoder, pi, d["in_dim"],goal_dim

def run(env_id, seeds, episodes, ckpt):
    encoder, pi,in_dim, goal_dim=load_policy(ckpt)
    device=th.device("cuda" if th.cuda.is_available() else "cpu")
    encoder.to(device)
    pi.to(device)
    stats=[]
    for sd in seeds:
        env=make_env(env_id, seed=sd)
        for _ in range(episodes):
            obs, info = env.reset(seed=sd)
            ret=0; done=False; steps=0
            while not done:
                obs_tensor, goal_tensor = obs_to_tensor(obs, goal_dim=goal_dim)
                obs_tensor = obs_tensor.unsqueeze(0).to(device) 
                with th.no_grad():
                    img_feat = encoder(obs_tensor)
                    if goal_tensor is not None:
                        goal_tensor = goal_tensor.unsqueeze(0) 
                        xt = torch.cat([img_feat, goal_tensor], dim=-1)
                    else:
                        xt = img_feat
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

