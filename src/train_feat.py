import argparse, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset import MVTecDataset, get_test_transform
from feature_model import FeatureExtractor, FeatureAE
from scoring import get_device

def main(a):
    device = get_device(); print("Using device:", device)
    os.makedirs("checkpoints_feat", exist_ok=True)
    train_full = MVTecDataset(a.root, a.category, "train")
    val_full = MVTecDataset(a.root, a.category, "train"); val_full.transform = get_test_transform()
    g = torch.Generator().manual_seed(42)             # same split as the pixel AE → fair comparison
    idx = torch.randperm(len(train_full), generator=g).tolist()
    n_val = max(1, int(0.15 * len(idx))); val_idx, train_idx = idx[:n_val], idx[n_val:]
    torch.save(val_idx, f"checkpoints_feat/{a.category}_val_idx.pt")
    tl = DataLoader(Subset(train_full, train_idx), a.bs, shuffle=True)
    vl = DataLoader(Subset(val_full, val_idx), a.bs)

    ext = FeatureExtractor().to(device)
    ae = FeatureAE(latent=a.latent).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=a.lr)
    best = float("inf")
    for ep in range(a.epochs):
        ae.train(); tr = 0
        for x in tl:
            f = ext(x.to(device)); loss = F.mse_loss(ae(f), f)
            opt.zero_grad(); loss.backward(); opt.step(); tr += loss.item()
        ae.eval(); v = 0
        with torch.no_grad():
            for x in vl:
                f = ext(x.to(device)); v += F.mse_loss(ae(f), f).item()
        v /= len(vl); tr /= len(tl)
        if v < best:
            best = v; torch.save(ae.state_dict(), f"checkpoints_feat/{a.category}.pt")
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d}  train {tr:.4f}  val {v:.4f}  best {best:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data"); p.add_argument("--category", default="bottle")
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--latent", type=int, default=200)
    main(p.parse_args())