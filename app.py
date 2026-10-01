import glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F, cv2, gradio as gr

sys.path.append("src")
from dataset import get_test_transform
from model import ConvAE
from feature_model import FeatureExtractor, FeatureAE, feature_anomaly_map
from patchcore import PatchCore
from scoring import get_device, reconstruct, error_maps, smooth, combine, image_score

device = get_device()
tf = get_test_transform()
CATS = sorted(os.path.basename(p)[:-3] for p in glob.glob("checkpoints_feat/*.pt")
              if "_val_idx" not in p)
_ext = FeatureExtractor().to(device)   # backbone loaded once, shared by every model
_idx = torch.load("checkpoints_feat/category_index.pth")
_emb, _lab = _idx["emb"].to(device), _idx["labels"].to(device)
_cache = {}

RESULTS = {"patchcore": "outputs_patchcore/{}/results.json",
           "feature": "outputs_feat/{}/results.json",
           "pixel": "outputs_pixel_baseline/{}/results.json"}


def img_auroc(path):
    try:
        r = json.load(open(path)); return r.get("combined", r)["img_auroc"]
    except FileNotFoundError:
        return -1.0


def best_model(cat):
    """The model with the highest test image AUROC for this category."""
    return max(RESULTS, key=lambda k: img_auroc(RESULTS[k].format(cat)))


def load(cat, kind):
    if (cat, kind) in _cache: return _cache[(cat, kind)]
    if kind == "patchcore":
        cal = json.load(open(f"checkpoints_patchcore/{cat}_calib.json"))
        bank = torch.load(f"checkpoints_patchcore/{cat}.pt", map_location="cpu").to(device)
        entry = (PatchCore(bank), cal["threshold"], None)
    elif kind == "feature":
        cal = json.load(open(f"checkpoints_feat/{cat}_calib.json"))
        m = FeatureAE(latent=cal["latent"]).to(device)
        m.load_state_dict(torch.load(f"checkpoints_feat/{cat}.pt", map_location=device))
        entry = (m.eval(), cal["threshold"], None)
    else:
        cal = json.load(open(f"checkpoints/{cat}_calib.json"))
        m = ConvAE(latent_ch=cal["latent"]).to(device)
        m.load_state_dict(torch.load(f"checkpoints/{cat}.pt", map_location=device))
        entry = (m.eval(), cal["thresholds"]["combined"], cal["stats"])
    _cache[(cat, kind)] = entry
    return entry


def detect_category(x):
    """Nearest training images by global feature vector (100% on the MVTec test set)."""
    with torch.no_grad():
        f = _ext(x.unsqueeze(0).to(device))
    q = F.normalize(f.mean(dim=(2, 3)), dim=1)
    top = (q @ _emb.T)[0].topk(5).indices
    return _idx["cats"][int(torch.mode(_lab[top].cpu()).values)]


def predict(img, cat, choice, scale):
    if img is None:
        return None, None, "Upload an image first."
    x = tf(img.convert("RGB"))
    detected = cat == "Auto-detect"
    if detected: cat = detect_category(x)
    kind = best_model(cat) if choice.startswith("Auto") else {"PatchCore": "patchcore", "Feature AE": "feature", "Pixel AE": "pixel"}[choice]
    model, thr, stats = load(cat, kind)

    t0 = time.perf_counter()
    if kind == "patchcore":
        amap = model.anomaly_map(_ext, x, device)
        score = float(amap.max())           # PatchCore scores by its worst patch
    elif kind == "feature":
        amap = feature_anomaly_map(_ext, model, x, device)
        score = image_score(amap)
    else:
        l2, se = error_maps(x, reconstruct(model, x, device))
        amap = smooth(combine(l2, se, stats))
        score = image_score(amap)
    ms = (time.perf_counter() - t0) * 1000

    t = thr * scale
    base = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Colour relative to the threshold (not per-image max), so normal images stay cool
    heat = cv2.applyColorMap((np.clip(amap / (2 * t), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), 0.45, 0)

    anomalous = score > t
    boxed = base.copy()
    if anomalous:
        region_t = max(t, 0.5 * amap.max())   # box only the strongest regions
        contours, _ = cv2.findContours((amap > region_t).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 30: continue
            x0, y0, w, h = cv2.boundingRect(c)
            cv2.rectangle(boxed, (x0, y0), (x0 + w, y0 + h), (255, 40, 40), 2)

    names = {"patchcore": "PatchCore", "feature": "feature autoencoder", "pixel": "pixel autoencoder"}
    verdict = "🔴 ANOMALY DETECTED" if anomalous else "🟢 NORMAL"
    text = (f"{verdict}\n\n" + (f"Detected category: {cat}\n" if detected else "") +
            f"Score: {score:.4f}   Threshold: {t:.4f}\n"
            f"Model: {names[kind]}   Inference: {ms:.0f} ms")
    return overlay, boxed, text


examples = [[f, os.path.basename(f).split("__")[0]] for f in sorted(glob.glob("examples/*.png"))]

with gr.Blocks(title="Industrial Anomaly Detection") as demo:
    gr.Markdown("# 🏭 Industrial Anomaly Detection\n"
                "Unsupervised defect detection on MVTec AD. The models were trained **only on "
                "defect-free images**, so anything that doesn't look normal gets flagged.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload an image")
            cat = gr.Dropdown(["Auto-detect"] + CATS, value="Auto-detect", label="Product category")
            choice = gr.Radio(["Auto (best per category)", "PatchCore", "Feature AE", "Pixel AE"],
                              value="Auto (best per category)", label="Model")
            scale = gr.Slider(0.5, 1.5, value=1.0, step=0.05,
                              label="Threshold scale (lower = more sensitive)")
            btn = gr.Button("Inspect", variant="primary")
        with gr.Column():
            out_heat = gr.Image(label="Anomaly heatmap")
            out_box = gr.Image(label="Flagged regions")
            out_txt = gr.Textbox(label="Result", lines=4)
    if examples:
        gr.Examples(examples, inputs=[inp, cat])
    args = [inp, cat, choice, scale]
    btn.click(predict, args, [out_heat, out_box, out_txt])
    scale.release(predict, args, [out_heat, out_box, out_txt])

demo.launch()