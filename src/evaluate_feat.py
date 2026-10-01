import argparse, json, os, numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from dataset import MVTecDataset, get_test_transform
from feature_model import FeatureExtractor, FeatureAE, feature_anomaly_map
from scoring import get_device, image_score

def main(a):
    device = get_device()
    ext = FeatureExtractor().to(device)
    ae = FeatureAE(latent=a.latent).to(device)
    ae.load_state_dict(torch.load(f"checkpoints_feat/{a.category}.pt", map_location=device)); ae.eval()

    # Threshold from held-out NORMAL images only
    val = MVTecDataset(a.root, a.category, "train"); val.transform = get_test_transform()
    val_idx = torch.load(f"checkpoints_feat/{a.category}_val_idx.pt")
    thr = float(np.percentile([image_score(feature_anomaly_map(ext, ae, val[i], device)) for i in val_idx], a.pct))

    test = MVTecDataset(a.root, a.category, "test")
    labels, scores, masks, maps, examples = [], [], [], [], {}
    for i in range(len(test)):
        x, y, mask = test[i]
        amap = feature_anomaly_map(ext, ae, x, device); s = image_score(amap)
        labels.append(y); scores.append(s); masks.append(mask[0].numpy().ravel()); maps.append(amap.ravel())
        d = test.images[i].split(os.sep)[-2]
        if d not in examples: examples[d] = (x, amap, mask[0], s)

    labels, scores = np.array(labels), np.array(scores); pred = (scores > thr).astype(int)
    res = {"img_auroc": roc_auc_score(labels, scores),
           "pix_auroc": roc_auc_score(np.concatenate(masks), np.concatenate(maps)),
           "precision": precision_score(labels, pred, zero_division=0),
           "recall": recall_score(labels, pred), "f1": f1_score(labels, pred)}
    print(f"{a.category:12s} img {res['img_auroc']:.3f}  pix {res['pix_auroc']:.3f}  "
          f"prec {res['precision']:.3f}  rec {res['recall']:.3f}  F1 {res['f1']:.3f}")

    out = f"outputs_feat/{a.category}"; os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/results.json", "w"), indent=2)
    json.dump({"threshold": thr, "latent": a.latent}, open(f"checkpoints_feat/{a.category}_calib.json", "w"), indent=2)

    fig, ax = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    for r, (d, (x, amap, gt, s)) in enumerate(examples.items()):
        img = x.permute(1, 2, 0).numpy()
        ax[r, 0].imshow(img); ax[r, 0].set_title(f"{d} ({s:.3f})")
        ax[r, 1].imshow(img); ax[r, 1].imshow(amap, cmap="jet", alpha=0.5); ax[r, 1].set_title("anomaly map")
        ax[r, 2].imshow(gt, cmap="gray"); ax[r, 2].set_title("ground truth")
        for c in range(3): ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/qualitative.png", dpi=110); plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--category", default="bottle")
    p.add_argument("--latent", type=int, default=200); p.add_argument("--pct", type=float, default=95)
    main(p.parse_args())