import argparse, os
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from pytorch_msssim import ssim
from dataset import MVTecDataset, get_test_transform
from model import ConvAE

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def loss_fn(x, r):
    return 0.5 * F.mse_loss(r, x) + 0.5 * (1 - ssim(r, x, data_range=1.0))

def main(args):
    device = get_device()
    print("Using device:", device)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(f"outputs/{args.category}", exist_ok=True)

    # Two copies of the same images: augmented for training, clean for validation
    train_full = MVTecDataset(args.root, args.category, "train", args.size)
    val_full = MVTecDataset(args.root, args.category, "train", args.size)
    val_full.transform = get_test_transform(args.size)

    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(train_full), generator=g).tolist()
    n_val = max(1, int(0.15 * len(idx)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    torch.save(val_idx, f"checkpoints/{args.category}_val_idx.pt")  # reused for thresholding later

    tl = DataLoader(Subset(train_full, train_idx), args.bs, shuffle=True, num_workers=0)
    vl = DataLoader(Subset(val_full, val_idx), args.bs, num_workers=0)

    model = ConvAE(latent_ch=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = float("inf")
    for ep in range(args.epochs):
        model.train(); tr = 0
        for x in tl:
            x = x.to(device)
            loss = loss_fn(x, model(x))
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item()
        sched.step()

        model.eval(); v = 0
        with torch.no_grad():
            for x in vl:
                x = x.to(device); v += loss_fn(x, model(x)).item()
        v /= len(vl); tr /= len(tl)

        if v < best:
            best = v
            torch.save(model.state_dict(), f"checkpoints/{args.category}.pt")
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  train {tr:.4f}  val {v:.4f}  best {best:.4f}")
            with torch.no_grad():
                x = next(iter(vl))[:4].to(device)
                save_image(torch.cat([x, model(x)]), f"outputs/{args.category}/epoch_{ep:03d}.png", nrow=4)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--category", default="bottle")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--size", type=int, default=256)
    main(p.parse_args())