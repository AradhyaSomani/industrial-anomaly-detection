import argparse, json, os
import numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from dataset import MVTecDataset, get_test_transform
from model import ConvAE
from scoring import get_device, reconstruct, error_maps, smooth, combine, image_score

VARIANTS = ["l2", "ssim", "combined"]

def all_maps(l2, se, stats):
    return {"l2": smooth(l2), "ssim": smooth(se), "combined": smooth(combine(l2, se, stats))}

def main(a):
    device = get_device()
    model = ConvAE(latent_ch=a.latent).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{a.category}.pt", map_location=device))
    model.eval()

    # --- 1. Calibrate on held-out NORMAL images only (no anomaly labels used) ---
    val = MVTecDataset(a.root, a.category, "train")
    val.transform = get_test_transform()
    val_idx = torch.load(f"checkpoints/{a.category}_val_idx.pt")
    val_maps = []
    for i in val_idx:
        x = val[i]
        val_maps.append(error_maps(x, reconstruct(model, x, device)))
    l2s = np.stack([m[0] for m in val_maps]); ses = np.stack([m[1] for m in val_maps])
    stats = {"l2_mean": float(l2s.mean()), "l2_std": float(l2s.std()),
             "ssim_mean": float(ses.mean()), "ssim_std": float(ses.std())}
    val_scores = {v: [] for v in VARIANTS}
    for l2, se in val_maps:
        m = all_maps(l2, se, stats)
        for v in VARIANTS: val_scores[v].append(image_score(m[v]))
    thresholds = {v: float(np.percentile(val_scores[v], a.pct)) for v in VARIANTS}

    # --- 2. Score the test set ---
    test = MVTecDataset(a.root, a.category, "test")
    labels, masks = [], []
    scores = {v: [] for v in VARIANTS}; pix = {v: [] for v in VARIANTS}
    examples = {}  # one example per defect type for the figure
    for i in range(len(test)):
        x, y, mask = test[i]
        r = reconstruct(model, x, device)
        m = all_maps(*error_maps(x, r), stats)
        labels.append(y); masks.append(mask[0].numpy().ravel())
        for v in VARIANTS:
            scores[v].append(image_score(m[v])); pix[v].append(m[v].ravel())
        defect = test.images[i].split(os.sep)[-2]
        if defect not in examples:
            examples[defect] = (x, r, m["combined"], mask[0], scores["combined"][-1])

    # --- 3. Metrics ---
    labels = np.array(labels); allmask = np.concatenate(masks)
    print(f"\n{a.category}: {len(labels)} test images, {labels.sum()} anomalous")
    print(f"{'variant':10s}{'img AUROC':>11s}{'pix AUROC':>11s}{'prec':>7s}{'recall':>8s}{'F1':>7s}")
    results = {}
    for v in VARIANTS:
        s = np.array(scores[v]); pred = (s > thresholds[v]).astype(int)
        res = {"img_auroc": roc_auc_score(labels, s),
               "pix_auroc": roc_auc_score(allmask, np.concatenate(pix[v])),
               "precision": precision_score(labels, pred, zero_division=0),
               "recall": recall_score(labels, pred), "f1": f1_score(labels, pred)}
        results[v] = res
        print(f"{v:10s}{res['img_auroc']:11.3f}{res['pix_auroc']:11.3f}"
              f"{res['precision']:7.3f}{res['recall']:8.3f}{res['f1']:7.3f}")

    # --- 4. Save calibration (the demo app uses this) and results ---
    json.dump({"stats": stats, "thresholds": thresholds, "latent": a.latent},
              open(f"checkpoints/{a.category}_calib.json", "w"), indent=2)
    os.makedirs(f"outputs/{a.category}", exist_ok=True)
    json.dump(results, open(f"outputs/{a.category}/results.json", "w"), indent=2)

    # --- 5. Qualitative figure: input | reconstruction | heatmap | ground truth ---
    n = len(examples)
    fig, ax = plt.subplots(n, 4, figsize=(12, 3 * n))
    for row, (defect, (x, r, amap, gt, sc)) in enumerate(examples.items()):
        img = x.permute(1, 2, 0).numpy()
        ax[row, 0].imshow(img); ax[row, 0].set_title(f"{defect} (score {sc:.2f})")
        ax[row, 1].imshow(r.permute(1, 2, 0).numpy().clip(0, 1)); ax[row, 1].set_title("reconstruction")
        ax[row, 2].imshow(img); ax[row, 2].imshow(amap, cmap="jet", alpha=0.5); ax[row, 2].set_title("anomaly map")
        ax[row, 3].imshow(gt, cmap="gray"); ax[row, 3].set_title("ground truth")
        for c in range(4): ax[row, c].axis("off")
    plt.tight_layout(); plt.savefig(f"outputs/{a.category}/qualitative.png", dpi=120)
    print(f"\nSaved outputs/{a.category}/qualitative.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--category", default="bottle")
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--pct", type=float, default=99)
    main(p.parse_args())