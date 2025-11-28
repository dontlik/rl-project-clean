import argparse, glob, os, numpy as np, torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
th.set_float32_matmul_precision("high")

class DemoSet(Dataset):
    def __init__(self, paths, obs_dim, goal_dim=0, take_every=1):
        X,Y,G=[],[],[]
        for p in paths:
            z=np.load(p, allow_pickle=True)
            obs=z["obs"]; act=z["action"]
            goal=z["goal"] if "goal" in z.files and goal_dim>0 else None
            T=len(act)
            for t in range(0,T,take_every):
                if goal is None:
                    G.append(np.zeros(goal_dim, dtype=np.float32))
                else:
                    G.append(goal[t])
                x=obs[t] 
                print(x.shape)
                X.append(x); Y.append(int(act[t]))
        self.X=th.tensor(np.array(X), dtype=th.float32)
        self.G=th.tensor(np.array(G), dtype=th.float32)
        self.Y=th.tensor(np.array(Y), dtype=th.long)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.X[i], self.G[i], self.Y[i]

class Cnn_Emb(nn.Module):
    def __init__(self, in_dim = (3,15,15), out_dim = 128):
        super().__init__()
        _, H, W = in_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_dim = 32 * H * W
        self.linear = nn.Linear(conv_out_dim, out_dim)
    def forward(self,x): 
        x = self.conv(x)      
        x = self.linear(x)    
        return x


class Policy(nn.Module):
    def __init__(self, n_actions, in_dim=128):
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
    #in_dim=cfg.obs_dim+cfg.goal_dim
    in_dim = 128 + cfg.goal_dim
    paths=sorted(glob.glob(cfg.demos_glob))
    assert paths, "no demos found"
    n=int(len(paths)*0.8)
    tr,va=paths[:n],paths[n:]
    trds, vads = DemoSet(tr,cfg.obs_dim,cfg.goal_dim), DemoSet(va,cfg.obs_dim,cfg.goal_dim)
    trld=DataLoader(trds,batch_size=cfg.batch,batch_sampler=None,shuffle=True)
    vald=DataLoader(vads,batch_size=cfg.batch,shuffle=False)
    
    device=th.device("cuda" if th.cuda.is_available() else "cpu")
    encoder = Cnn_Emb(in_dim=(3,15,15), out_dim=128).to(device)

    pi=Policy(cfg.n_actions,in_dim=128 + cfg.goal_dim).to(device)
    opt=th.optim.Adam(pi.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    ce=nn.CrossEntropyLoss()
    best=float("inf")
    os.makedirs(os.path.dirname(cfg.ckpt), exist_ok=True)

    for ep in range(cfg.epochs):
        pi.train(); tl=0
        for obs, goal, action in tqdm(trld, desc=f"epoch {ep}"):
            obs, goal, action=obs.to(device), goal.to(device), action.to(device)
            x = encoder(obs)
            x = th.cat([x, goal], dim=-1)  
            logits=pi(x); loss=ce(logits,action)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()*len(x)
        tl/=len(trds)

        pi.eval(); vl=0; acc=0; n=0
        with th.no_grad():
            for x, g, y in vald: 
                x, g, y = x.to(device), g.to(device), y.to(device)
                x = encoder(x)
                x = th.cat([x, g], dim=-1) 
                logits=pi(x); loss=ce(logits,y); vl+=loss.item()*len(x)
                pred=logits.argmax(-1); acc+=(pred==y).sum().item(); n+=len(x)
        vl/=len(vads); acc/=max(1,n)
        print(f"[ep {ep}] train_ce={tl:.4f} val_ce={vl:.4f} val_acc={acc:.3f}")
        if vl<best: best=vl; th.save({"state_dict":pi.state_dict(),"encoder": encoder.state_dict(),"in_dim":in_dim,"goal_dim": cfg.goal_dim,}, cfg.ckpt)
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
    obs_dim=z["obs"].shape[1:]; print("inferred obs_dim:",obs_dim)
    args.obs_dim=obs_dim
    train(args)

