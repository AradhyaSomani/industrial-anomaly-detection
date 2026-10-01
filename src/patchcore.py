"""PatchCore (Roth et al., CVPR 2022) on the same frozen WideResNet-50 features as the feature autoencoder.

Memory bank = feature vectors of every patch in the normal training images, reduced to a
representative subset by greedy coreset selection. A test patch's anomaly score is the distance
to its nearest neighbour in the bank; the image score is the highest patch score.
No training. Uses the same seeded validation split and 95th-percentile threshold as the other models.

    python -u src/patchcore.py --category bottle
"""
import argparse, json, os, time, numpy as np, torch, torch.nn.functional as F, cv2
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from dataset import MVTecDataset, get_test_transform
from feature_model import FeatureExtractor
from scoring import get_device


def patch_features(ext, x, device):
    """One image -> (H*W) x C patch feature vectors, plus the feature-map size."""
    with torch.no_grad():
        f = ext(x.unsqueeze(0).to(device))               # 1 x 1536 x H x W
    c, h, w = f.shape[1:]
    return f[0].reshape(c, h * w).T, (h, w)


def greedy_coreset(feats, ratio, device, proj_dim=128, seed=0):
    """Pick ratio * N patches that cover the feature space (iterative farthest-point selection).
    Distances are computed on a random 128-d projection to keep it fast."""
    n = feats.shape[0]
    k = max(1, int(n * ratio))
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(feats.shape[1], proj_dim, generator=g).to(device)
    proj = torch.cat([feats[i:i + 50000].to(device) @ R for i in range(0, n, 50000)])
    selected = torch.zeros(k, dtype=torch.long, device=device)
    min_d = torch.full((n,), float("inf"), device=device)
    last = torch.randint(n, (1,), generator=g).to(device)
    for i in range(k):
        selected[i] = last[0]
        min_d = torch.minimum(min_d, ((proj - proj[last]) ** 2).sum(1))
        last = torch.argmax(min_d).unsqueeze(0)
    return selected.cpu()


class PatchCore:
    def __init__(self, bank):
        self.bank = bank                                  # M x C on device
        self.bank_sq = (bank ** 2).sum(1)

    def anomaly_map(self, ext, x, device, out_size=256, sigma=4):
        q, (h, w) = patch_features(ext, x, device)
        d2 = (q ** 2).sum(1, keepdim=True) + self.bank_sq[None, :] - 2 * q @ self.bank.T
        s = d2.min(1).values.clamp_min(0).sqrt().reshape(1, 1, h, w)
        s = F.interpolate(s, size=out_size, mode="bilinear", align_corners=False)
        return cv2.GaussianBlur(s[0, 0].cpu().numpy(), (0, 0), sigma)


def main(a):
    device = get_device()
    ext = FeatureExtractor().to(device)

    # Same seeded split as the autoencoders: validation images never enter the memory bank
    train = MVTecDataset(a.root, a.category, "train")
    train.transform = get_test_transform()               # no augmentation for PatchCore
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(train), generator=g).tolist()
    n_val = max(1, int(0.15 * len(idx)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # --- 1. Memory bank ---
    t0 = time.perf_counter()
    feats = torch.cat([patch_features(ext, train[i], device)[0].cpu() for i in train_idx])
    sel = greedy_coreset(feats, a.coreset, device)
    bank = feats[sel].to(device)
    del feats
    model = PatchCore(bank)
    print(f"{a.category}: memory bank {bank.shape[0]} patches from {len(train_idx)} images "
          f"({a.coreset:.0%} coreset), built in {time.perf_counter() - t0:.0f} s")

    # --- 2. Threshold from held-out NORMAL images only ---
    val_scores = [float(model.anomaly_map(ext, train[i], device).max()) for i in val_idx]
    thr = float(np.percentile(val_scores, a.pct))

    # Save the bank and threshold so the demo app can use PatchCore
    os.makedirs("checkpoints_patchcore", exist_ok=True)
    torch.save(bank.cpu(), f"checkpoints_patchcore/{a.category}.pt")
    json.dump({"threshold": thr, "coreset": a.coreset, "bank_size": int(bank.shape[0])},
              open(f"checkpoints_patchcore/{a.category}_calib.json", "w"), indent=2)

    # --- 3. Test set ---
    test = MVTecDataset(a.root, a.category, "test")
    labels, scores, masks, maps, times = [], [], [], [], []
    for i in range(len(test)):
        x, y, mask = test[i]
        t1 = time.perf_counter()
        amap = model.anomaly_map(ext, x, device)
        times.append((time.perf_counter() - t1) * 1000)
        labels.append(y); scores.append(float(amap.max()))
        masks.append(mask[0].numpy().ravel()); maps.append(amap.ravel())

    labels, scores = np.array(labels), np.array(scores)
    pred = (scores > thr).astype(int)
    res = {"img_auroc": roc_auc_score(labels, scores),
           "pix_auroc": roc_auc_score(np.concatenate(masks), np.concatenate(maps)),
           "precision": precision_score(labels, pred, zero_division=0),
           "recall": recall_score(labels, pred),
           "f1": f1_score(labels, pred),
           "ms_per_image": float(np.median(times)),
           "bank_size": int(bank.shape[0])}
    print(f"{a.category:12s} img {res['img_auroc']:.3f}  pix {res['pix_auroc']:.3f}  "
          f"prec {res['precision']:.3f}  rec {res['recall']:.3f}  F1 {res['f1']:.3f}  "
          f"{res['ms_per_image']:.0f} ms/img")
    out = f"outputs_patchcore/{a.category}"
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/results.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--category", default="bottle")
    p.add_argument("--coreset", type=float, default=0.01)
    p.add_argument("--pct", type=float, default=95)
    main(p.parse_args())