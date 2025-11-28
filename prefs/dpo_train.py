import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse, torch as th, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import Dataset, DataLoader
from bc.train_bc import Policy, Cnn_Emb
from tqdm import tqdm

class PairSet(Dataset):
    def __init__(self, pairs):
        self.pairs=pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        P=self.pairs[i]["pos"]; N=self.pairs[i]["neg"]
        # pack into tensors
        obs_p=th.tensor(P["obs"], dtype=th.float32)
        act_p=th.tensor(P["action"], dtype=th.long)
        obs_n=th.tensor(N["obs"], dtype=th.float32)
        act_n=th.tensor(N["action"], dtype=th.long)
        return obs_p,act_p,obs_n,act_n

def seq_logprob(pi, obs_seq, act_seq):
    # obs_seq: [L, in_dim] ; act_seq: [L]
    logits = pi(obs_seq)                 # [L, n_actions]
    logp   = nn.functional.log_softmax(logits, dim=-1)
    lp     = logp.gather(-1, act_seq.unsqueeze(-1)).squeeze(-1)  # [L]
    return lp.mean()                      # length-normalized

def dpo_step(pi_theta, pi_ref, batch, beta, device):
    obs_p, act_p, obs_n, act_n = [x.to(device) for x in batch]
    B = obs_p.shape[0]
    vals=[]
    with th.no_grad():
        lp_ref_p = th.stack([seq_logprob(pi_ref, obs_p[i], act_p[i]) for i in range(B)])  # [B]
        lp_ref_n = th.stack([seq_logprob(pi_ref, obs_n[i], act_n[i]) for i in range(B)])
    lp_theta_p  = th.stack([seq_logprob(pi_theta, obs_p[i], act_p[i]) for i in range(B)])
    lp_theta_n  = th.stack([seq_logprob(pi_theta, obs_n[i], act_n[i]) for i in range(B)])

    margin = beta * ((lp_theta_p - lp_theta_n) - (lp_ref_p - lp_ref_n))  # [B]
    loss = nn.functional.binary_cross_entropy_with_logits(margin, th.ones_like(margin))
    return loss, {
        "lp_theta_p": lp_theta_p.mean().item(),
        "lp_theta_n": lp_theta_n.mean().item(),
        "lp_ref_p": lp_ref_p.mean().item(),
        "lp_ref_n": lp_ref_n.mean().item(),
    }

def approx_kl(pi_theta, pi_ref, batch, device):
    # simple Monte-Carlo KL on sequences in batch
    obs_p, act_p, _, _ = [x.to(device) for x in batch]
    with th.no_grad():
        logits_t = pi_theta(obs_p.view(-1, obs_p.shape[-1]))
        logits_r = pi_ref(obs_p.view(-1, obs_p.shape[-1]))
        logp_t = nn.functional.log_softmax(logits_t, -1)
        logp_r = nn.functional.log_softmax(logits_r, -1)
        p_t = logp_t.exp()
        kl = (p_t*(logp_t - logp_r)).sum(-1).mean().item()
    return kl

def main(pairs_path, ref_ckpt, beta=0.5, lr=1e-4, batch_size=8, epochs=5, save_path="prefs/checkpoints/dpo.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device=th.device("cuda" if th.cuda.is_available() else "cpu")
    pairs=th.load(pairs_path,weights_only=False)
    ds=PairSet(pairs)
    dl=DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    d = th.load(ref_ckpt, map_location="cpu")

    in_dim = d["in_dim"]
    n_actions = 7  # rough infer; DoorKey=7 anyway

    print("Input dimension (in_dim):", in_dim)
    print("Number of actions (n_actions):", n_actions)

    goal_dim = d["goal_dim"]
    cnn_out_dim = in_dim - goal_dim

    pi_ref = Policy(n_actions, in_dim); pi_ref.load_state_dict(th.load(ref_ckpt, map_location="cpu")["state_dict"])
    encoder_ref = Cnn_Emb(in_dim=(3, 15, 15), out_dim=cnn_out_dim); encoder_ref.load_state_dict(d["encoder"])
    for p in pi_ref.parameters(): p.requires_grad=False
    pi_ref.eval(); pi_ref.to(device)
    encoder_ref.eval(); encoder_ref.to(device)

    encoder_theta = Cnn_Emb(in_dim=(3, 15, 15), out_dim=cnn_out_dim); encoder_theta.load_state_dict(d["encoder"])
    pi_theta = Policy(n_actions, in_dim); pi_theta.load_state_dict(th.load(ref_ckpt, map_location="cpu")["state_dict"])
    pi_theta.to(device)
    encoder_theta.to(device)
    opt=optim.Adam(pi_theta.parameters(), lr=lr)

    for ep in range(epochs):
        pi_theta.train(); tot=0; kls=[]
        for batch in tqdm(dl, desc=f"DPO ep{ep}"):
            opt.zero_grad()
            loss, logs = dpo_step(pi_theta, pi_ref, batch, beta, device)
            loss.backward(); opt.step()
            tot += loss.item()*batch_size
            kls.append(approx_kl(pi_theta, pi_ref, batch, device))
        print(f"[ep {ep}] dpo_loss={tot/len(ds):.4f}  MC-KL≈{np.mean(kls):.4f}")

    th.save({"state_dict":pi_theta.state_dict(), "in_dim":in_dim, "encoder": encoder_theta.state_dict(), "goal_dim":goal_dim}, save_path)
    print("saved:", save_path)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs",type=str,default="data/prefs/pairs.pt")
    ap.add_argument("--ref_ckpt",type=str,default="bc/checkpoints/pi_ref.pt")
    ap.add_argument("--beta",type=float,default=0.5)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--batch_size",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--save_path",type=str,default="prefs/checkpoints/dpo.pt")
    args=ap.parse_args()
    main(args.pairs, args.ref_ckpt, args.beta, args.lr, args.batch_size, args.epochs, args.save_path)
