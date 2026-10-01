import numpy as np, torch, cv2
from skimage.metrics import structural_similarity

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

@torch.no_grad()
def reconstruct(model, x, device):
    return model(x.unsqueeze(0).to(device))[0].cpu()

def error_maps(x, r):
    xn, rn = x.permute(1, 2, 0).numpy(), r.permute(1, 2, 0).numpy()
    l2 = ((xn - rn) ** 2).mean(axis=2)
    _, s = structural_similarity(xn, rn, channel_axis=2, data_range=1.0, full=True, win_size=11)
    return l2, 1 - s.mean(axis=2)

def smooth(m, sigma=4):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)

def combine(l2, ssim_err, stats):
    # z-normalise each error type with stats from normal validation images
    return (0.5 * (l2 - stats["l2_mean"]) / stats["l2_std"]
            + 0.5 * (ssim_err - stats["ssim_mean"]) / stats["ssim_std"])

def image_score(amap, top_k=0.01):
    flat = np.sort(amap.ravel())[::-1]
    return float(flat[: max(1, int(len(flat) * top_k))].mean())