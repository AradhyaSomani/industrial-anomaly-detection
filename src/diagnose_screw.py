import os, numpy as np, torch, matplotlib.pyplot as plt
from dataset import MVTecDataset
from feature_model import FeatureExtractor, FeatureAE, feature_anomaly_map
from scoring import get_device, image_score

device = get_device()
ext = FeatureExtractor().to(device)
ae = FeatureAE(latent=200).to(device)
ae.load_state_dict(torch.load("checkpoints_feat/screw.pt", map_location=device)); ae.eval()

test = MVTecDataset("data", "screw", "test")
rows = []
for i in range(len(test)):
    x, y, _ = test[i]
    amap = feature_anomaly_map(ext, ae, x, device)
    rows.append((test.images[i].split(os.sep)[-2], image_score(amap), x, amap))

print("Mean score by type:")
for t in sorted({r[0] for r in rows}):
    s = [r[1] for r in rows if r[0] == t]
    print(f"  {t:18s} n={len(s):3d}  mean {np.mean(s):.5f}  min {np.min(s):.5f}  max {np.max(s):.5f}")

good = sorted([r for r in rows if r[0] == "good"], key=lambda r: -r[1])[:6]
fig, ax = plt.subplots(2, 6, figsize=(18, 6))
for j, (_, s, x, amap) in enumerate(good):
    img = x.permute(1, 2, 0).numpy()
    ax[0, j].imshow(img); ax[0, j].set_title(f"good, score {s:.4f}")
    ax[1, j].imshow(img); ax[1, j].imshow(amap, cmap="jet", alpha=0.5)
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.tight_layout(); plt.savefig("outputs_feat/screw_diagnosis.png", dpi=100)
print("Saved outputs_feat/screw_diagnosis.png")