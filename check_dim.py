import torch

ckpt_path = "bc/checkpoints/pi_ref.pt"
d = torch.load(ckpt_path, map_location="cpu")

if "goal_dim" in d:
    print(f"✅ goal_dim found: {d['goal_dim']}")
else:
    print("❌ goal_dim not found in checkpoint!")
