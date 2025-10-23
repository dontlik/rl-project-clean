import argparse, glob, os, numpy as np, torch as th, random

def segment_traj(tr, L, stride):
    T=len(tr["action"]); segs=[]
    for t0 in range(0, T-L+1, stride):
        t1=t0+L
        segs.append({
            "obs": tr["obs"][t0:t1],
            "action": tr["action"][t0:t1],
            "goal": tr["goal"][t0:t1] if "goal" in tr.files else None,
            "reward": tr["reward"][t0:t1],
        })
    return segs

def score(seg, method="return"):
    if method=="return": return float(np.sum(seg["reward"]))
    # you can add heuristics (e.g., distance to goal)
    return 0.0

def main(globpat, out_path, L=20, stride=10, pairs_per_traj=20):
    paths=sorted(glob.glob(globpat)); assert paths
    pairs=[]
    rng=random.Random(0)
    for p in paths:
        z=np.load(p, allow_pickle=True)
        segs=segment_traj(z, L, stride)
        if len(segs)<2: continue
        scores=[score(s) for s in segs]
        used=set()
        K=min(pairs_per_traj, len(segs)*(len(segs)-1)//2)
        while len(used)<K:
            i,j=rng.randrange(len(segs)), rng.randrange(len(segs))
            if i==j: continue
            key=tuple(sorted((i,j)))
            if key in used: continue
            used.add(key)
            si,sj=scores[i],scores[j]
            if si==sj: continue
            pos,neg=(segs[i],segs[j]) if si>sj else (segs[j],segs[i])
            pairs.append({"pos":pos, "neg":neg, "meta":{"sp":float(max(si,sj)),"sn":float(min(si,sj)),"src":p}})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    th.save(pairs, out_path)
    print(f"saved {len(pairs)} pairs to {out_path}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob",type=str, default="data/demos/MiniGrid-DoorKey-6x6-v0/*.npz")
    ap.add_argument("--out",type=str, default="data/prefs/pairs.pt")
    ap.add_argument("--L",type=int, default=20)
    ap.add_argument("--stride",type=int, default=10)
    ap.add_argument("--pairs_per_traj",type=int, default=20)
    args=ap.parse_args()
    main(args.glob, args.out, args.L, args.stride, args.pairs_per_traj)

