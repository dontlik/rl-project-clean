# RL-Project

# BC → DPO on MiniGrid (DoorKey)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

```
## 1) Generate expert demos
```bash
python utils/expert_demos_best.py --env_id MiniGrid-DoorKey-6x6-v0 --seeds 0 1 --episodes_per_seed 15
```

## 2) Train Behavior Cloning (π_ref)
```bash
python bc/train_bc.py --demos_glob "data/demos/MiniGrid-DoorKey-6x6-v0/*.npz"
python -m bc.eval_bc --ckpt bc/checkpoints/pi_ref.pt --episodes 60
```

## 3) Preference pairs (synthetic)
```bash
python prefs/make_pairs.py --glob "data/demos/MiniGrid-DoorKey-6x6-v0/*.npz" --out data/prefs/pairs.pt --L 20 --stride 10
```

## 4) DPO fine-tuning
```bash
python prefs/dpo_train.py --pairs data/prefs/pairs.pt --ref_ckpt bc/checkpoints/pi_ref.pt --beta 0.5 --epochs 5
python bc/eval_bc.py --ckpt prefs/checkpoints/dpo.pt --episodes 60
```

---

## Notes & next steps

- If your **obs_dim** inference in `train_bc.py` prints a surprising value, open one `.npz` and check `obs.shape[-1]`.  
- For **new goals/environments**, just change `env_id` (e.g., `MiniGrid-MultiRoom-N2-S4-v0`), regenerate demos/pairs, and run DPO again.  
- To add **human preferences**, build a tiny Gradio UI that loads two video clips or GIFs of segments and writes pairs into `data/prefs/*.pt`—the `dpo_train.py` will consume them unchanged.

If you paste in these files and run the commands in order, you’ll have a working baseline from **Week 1 → Week 2 (DPO)**. If you want, I can also provide a small **plot script** to visualize *success vs #pairs* and *KL vs steps* for your report.
::contentReference[oaicite:0]{index=0}
