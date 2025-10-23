import argparse, glob, os, numpy as np, torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
th.set_float32_matmul_precision("high")

class DemoSet(Dataset):
    def __init__(self, paths, obs_dim, goal_dim=0, take_every=1):
        X,Y=[],[]
        for p in paths:
            z=np.load(p, allow_pickle=True)
            obs=z["obs"]; act=z["action"]
            goal=z["goal"] if "goal" in z.files and goal_dim>0 else None
            T=len(act)
            for t in range(0,T,take_every):
                x=obs[t] if goal is None else np.concatenate([obs[t], goal[t]],-1)
                X.append(x); Y.append(int(act[t]))
        self.X=th.tensor(np.array(X), dtype=th.float32)
        self.Y=th.tensor(np.array(Y), dtype=th.long)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.X[i], self.Y[i]

class Policy(nn.Module):
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,256), nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(256,128), nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(128,n_actions)
        )
    def forward(self,x): return self.net(x)
    def log_prob(self, obs, act):
        logits=self(obs); logp=nn.functional.log_softmax(logits,dim=-1)
        return logp.gather(-1, act.unsqueeze(-1)).squeeze(-1)

def train(cfg):
    th.manual_seed(cfg.seed)
    in_dim=cfg.obs_dim+cfg.goal_dim
    paths=sorted(glob.glob(cfg.demos_glob))
    assert paths, "no demos found"
    n=int(len(paths)*0.8)
    tr,va=paths[:n],paths[n:]
    trds, vads = DemoSet(tr,cfg.obs_dim,cfg.goal_dim), DemoSet(va,cfg.obs_dim,cfg.goal_dim)
    trld=DataLoader(trds,batch_size=cfg.batch,batch_sampler=None,shuffle=True)
    vald=DataLoader(vads,batch_size=cfg.batch,shuffle=False)
    device=th.device("cuda" if th.cuda.is_available() else "cpu")
    pi=Policy(in_dim,cfg.n_actions).to(device)
    opt=th.optim.Adam(pi.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    ce=nn.CrossEntropyLoss()
    best=float("inf")
    os.makedirs(os.path.dirname(cfg.ckpt), exist_ok=True)

    for ep in range(cfg.epochs):
        pi.train(); tl=0
        for x,y in tqdm(trld, desc=f"epoch {ep}"):
            x,y=x.to(device),y.to(device)
            logits=pi(x); loss=ce(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()*len(x)
        tl/=len(trds)

        pi.eval(); vl=0; acc=0; n=0
        with th.no_grad():
            for x,y in vald:
                x,y=x.to(device),y.to(device)
                logits=pi(x); loss=ce(logits,y); vl+=loss.item()*len(x)
                pred=logits.argmax(-1); acc+=(pred==y).sum().item(); n+=len(x)
        vl/=len(vads); acc/=max(1,n)
        print(f"[ep {ep}] train_ce={tl:.4f} val_ce={vl:.4f} val_acc={acc:.3f}")
        if vl<best: best=vl; th.save({"state_dict":pi.state_dict(),"in_dim":in_dim}, cfg.ckpt)
    print("saved:", cfg.ckpt)

if __name__=="__main__":
    import dataclasses
    @dataclasses.dataclass
    class C:
        demos_glob = "data/demos/MiniGrid-DoorKey-6x6-v0/*.npz"
        obs_dim = 147
        goal_dim = 0
        n_actions = 7
        # batch = 256
        batch = 12
        lr = 3e-4
        # wd = 1e-4
        wd = 1e-9
        # epochs = 30
        epochs = 500
        seed = 0
        ckpt = "bc/checkpoints/pi_ref.pt"
    # NOTE: set obs_dim to the flattened size: for FullyObsWrapper, DoorKey-6x6 -> image shape [height,width,3].
    # At runtime we can infer. Let's infer from a sample file:
    ap=argparse.ArgumentParser()
    ap.add_argument("--demos_glob",type=str,default=C.demos_glob)
    ap.add_argument("--goal_dim",type=int,default=C.goal_dim)
    ap.add_argument("--n_actions",type=int,default=C.n_actions)
    ap.add_argument("--batch",type=int,default=C.batch)
    ap.add_argument("--lr",type=float,default=C.lr)
    ap.add_argument("--wd",type=float,default=C.wd)
    ap.add_argument("--epochs",type=int,default=C.epochs)
    ap.add_argument("--seed",type=int,default=C.seed)
    ap.add_argument("--ckpt",type=str,default=C.ckpt)
    args=ap.parse_args()

    # infer obs_dim from a file
    import glob as G
    f=sorted(G.glob(args.demos_glob))[0]; z=np.load(f,allow_pickle=True)
    obs_dim=z["obs"].shape[-1]; print("inferred obs_dim:",obs_dim)
    args.obs_dim=obs_dim
    train(args)

