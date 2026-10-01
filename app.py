import glob, json, os, sys, time
import numpy as np, torch, cv2, gradio as gr

sys.path.append("src")
from dataset import get_test_transform
from model import ConvAE
from feature_model import FeatureExtractor, FeatureAE, feature_anomaly_map, map_from_features
from scoring import get_device, reconstruct, error_maps, smooth, combine, image_score

device = get_device()
tf = get_test_transform()
CATS = sorted(os.path.basename(p)[:-3] for p in glob.glob("checkpoints_feat/*.pt")
              if "_val_idx" not in p)
_ext = FeatureExtractor().to(device)   # load the backbone once at startup
_cache = {}

def img_auroc(path):
    try:
        r = json.load(open(path)); return r.get("combined", r)["img_auroc"]
    except FileNotFoundError:
        return -1.0

def best_model(cat):
    feat = img_auroc(f"outputs_feat/{cat}/results.json")
    pix = img_auroc(f"outputs_pixel_baseline/{cat}/results.json")
    return "feature" if feat >= pix else "pixel"

def load(cat, kind):
    if (cat, kind) in _cache: return _cache[(cat, kind)]
    if kind == "feature":
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
    with torch.no_grad():
        f = _ext(x.unsqueeze(0).to(device))   # backbone runs once, reused for every category
    best, best_ratio = None, float("inf")
    for c in CATS:
        ae, thr, _ = load(c, "feature")
        ratio = image_score(map_from_features(f, ae)) / thr
        if ratio < best_ratio: best, best_ratio = c, ratio
    return best

def predict(img, cat, choice, scale):
    if img is None:
        return None, None, "Upload an image first."
    x = tf(img.convert("RGB"))
    detected = cat == "Auto-detect"
    if detected: cat = detect_category(x)
    kind = best_model(cat) if choice.startswith("Auto") else choice.lower()
    model, thr, stats = load(cat, kind)

    t0 = time.perf_counter()
    if kind == "feature":
        amap = feature_anomaly_map(_ext, model, x, device)
    else:
        l2, se = error_maps(x, reconstruct(model, x, device))
        amap = smooth(combine(l2, se, stats))
    ms = (time.perf_counter() - t0) * 1000

    score, t = image_score(amap), thr * scale
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

    verdict = "🔴 ANOMALY DETECTED" if anomalous else "🟢 NORMAL"
    text = (f"{verdict}\n\n" + (f"Detected category: {cat}\n" if detected else "") +
            f"Score: {score:.4f}   Threshold: {t:.4f}\n"
            f"Model: {kind} autoencoder   Inference: {ms:.0f} ms")
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
            choice = gr.Radio(["Auto (best per category)", "Feature", "Pixel"],
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

demo.launch(share=True)