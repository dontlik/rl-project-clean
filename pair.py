# -*- coding: utf-8 -*-


import torch

path = "data/prefs/pairs.pt"
data = torch.load(path, weights_only=False)

print(f"type: {type(data)}")
print(f"length: {len(data)}")

for i, item in enumerate(data[:5]):
    print(f"\n--- Pair {i} ---")
    if isinstance(item, dict) and "pos" in item and "neg" in item:
        pos_obs = item["pos"]["obs"]
        neg_obs = item["neg"]["obs"]
        pos_action = item["pos"]["action"]
        neg_action = item["neg"]["action"]

        print(f"pos_obs: shape={pos_obs.shape}, dtype={pos_obs.dtype}")
        print(f"neg_obs: shape={neg_obs.shape}, dtype={neg_obs.dtype}")
        print(f"pos_action: {pos_action}, neg_action: {neg_action}")
    else:
        print("Unknown format:", type(item))