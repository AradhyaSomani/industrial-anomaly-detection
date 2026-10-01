"""Build a lookup index for automatic category detection.

Each training image (normal only) is reduced to one global feature vector:
the WideResNet-50 feature map averaged over all positions, then L2-normalised.
A new image is assigned to the category of its nearest training images.
Defects are local, so they barely change this global vector, unlike the
anomaly score, which a defect raises on purpose.

Saved as .pth (not .pt) so the app does not mistake it for a category model.
"""
import glob, os, torch, torch.nn.functional as F
from dataset import MVTecDataset, get_test_transform
from feature_model import FeatureExtractor
from scoring import get_device


def main(root="data"):
    device = get_device()
    ext = FeatureExtractor().to(device)
    cats = sorted(os.path.basename(p)[:-3] for p in glob.glob("checkpoints_feat/*.pt")
                  if "_val_idx" not in p)
    embs, labels = [], []
    for ci, c in enumerate(cats):
        ds = MVTecDataset(root, c, "train")
        ds.transform = get_test_transform()          # no augmentation
        for i in range(len(ds)):
            with torch.no_grad():
                f = ext(ds[i].unsqueeze(0).to(device)).mean(dim=(2, 3))   # 1 x 1536
            embs.append(F.normalize(f, dim=1).cpu())
            labels.append(ci)
        print(f"{c:12s} {len(ds)} training images")
    torch.save({"cats": cats, "emb": torch.cat(embs), "labels": torch.tensor(labels)},
               "checkpoints_feat/category_index.pth")
    print(f"Saved checkpoints_feat/category_index.pth ({len(labels)} images, {len(cats)} categories)")


if __name__ == "__main__":
    main()
