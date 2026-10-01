"""Measure automatic category detection on every MVTec test image.

--method ratio : category where the image looks most normal (lowest score / threshold). Original demo method.
--method knn   : nearest training images by global feature vector (needs build_category_index.py first).
"""
import argparse, glob, json, os, torch, torch.nn.functional as F
from dataset import MVTecDataset
from feature_model import FeatureExtractor, FeatureAE, map_from_features
from scoring import get_device, image_score


def main(a):
    device = get_device()
    ext = FeatureExtractor().to(device)

    if a.method == "ratio":
        cats = sorted(os.path.basename(p)[:-3] for p in glob.glob("checkpoints_feat/*.pt")
                      if "_val_idx" not in p)
        models = {}
        for c in cats:
            cal = json.load(open(f"checkpoints_feat/{c}_calib.json"))
            ae = FeatureAE(latent=cal["latent"]).to(device)
            ae.load_state_dict(torch.load(f"checkpoints_feat/{c}.pt", map_location=device))
            models[c] = (ae.eval(), cal["threshold"])

        def predict(f):
            ratios = {c: image_score(map_from_features(f, ae)) / thr for c, (ae, thr) in models.items()}
            return min(ratios, key=ratios.get)
    else:
        idx = torch.load("checkpoints_feat/category_index.pth")
        cats, emb, lab = idx["cats"], idx["emb"].to(device), idx["labels"].to(device)

        def predict(f):
            q = F.normalize(f.mean(dim=(2, 3)), dim=1)
            top = (q @ emb.T)[0].topk(a.k).indices
            return cats[int(torch.mode(lab[top].cpu()).values)]
    stats = {c: {"good": [0, 0], "defect": [0, 0]} for c in cats}   # [correct, total]
    confusions = {}
    for true_cat in cats:
        test = MVTecDataset(a.root, true_cat, "test")
        for i in range(len(test)):
            x, y, _ = test[i]
            with torch.no_grad():
                f = ext(x.unsqueeze(0).to(device))
            pred = predict(f)
            kind = "defect" if y else "good"
            stats[true_cat][kind][0] += int(pred == true_cat)
            stats[true_cat][kind][1] += 1
            if pred != true_cat:
                key = f"{true_cat} -> {pred}"
                confusions[key] = confusions.get(key, 0) + 1
        g, d = stats[true_cat]["good"], stats[true_cat]["defect"]
        print(f"{true_cat:12s} good {g[0]}/{g[1]}   defective {d[0]}/{d[1]}")

    tg = sum(s["good"][0] for s in stats.values()); ng = sum(s["good"][1] for s in stats.values())
    td = sum(s["defect"][0] for s in stats.values()); nd = sum(s["defect"][1] for s in stats.values())
    print(f"\n[{a.method}] Overall accuracy: {(tg + td) / (ng + nd):.3%}  "
          f"(good {tg / ng:.3%}, defective {td / nd:.3%}, {ng + nd} images)")
    if confusions:
        print("Misrouted:", ", ".join(f"{k} ({v})" for k, v in sorted(confusions.items(), key=lambda kv: -kv[1])))
    os.makedirs("outputs_feat", exist_ok=True)
    json.dump({"method": a.method, "per_category": stats, "confusions": confusions},
              open(f"outputs_feat/autodetect_{a.method}.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--method", choices=["ratio", "knn"], default="knn")
    p.add_argument("--k", type=int, default=5)
    main(p.parse_args())
