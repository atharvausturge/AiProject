import io
import json
import os
import re
import sys

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

sys.path.insert(0, os.path.dirname(__file__))
from dataset import eval_tf, pick_device  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "models", "clickbait_cnn.pt")


@st.cache_resource
def load_model():
    m = models.mobilenet_v2()
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 2))
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


@st.cache_data
def load_metrics():
    p = MODEL_PATH.replace(".pt", "_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def extract_video_id(url):
    """Pull the 11-char video ID out of any common YouTube URL form."""
    patterns = [
        r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",  # raw id
    ]
    for pat in patterns:
        m = re.search(pat, url.strip())
        if m:
            return m.group(1)
    return None


def fetch_thumbnail(video_id):
    r = requests.get(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg", timeout=15)
    if r.status_code == 200 and len(r.content) > 2000:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return None


def verdict(score):
    if score >= 0.66:
        return "Looks like clickbait", ":red"
    if score >= 0.40:
        return "Borderline", ":orange"
    return "Looks clean", ":green"


st.set_page_config(page_title="Clickbait Thumbnail Detector", page_icon="mag")
st.title("Clickbait Thumbnail Detector")
st.caption(
    "A convolutional neural network (transfer-learned from ImageNet) scores how "
    "*clickbait* a YouTube thumbnail looks, and shows a heatmap of where it looked."
)

if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python src/train_cnn.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()
device = pick_device()
model.to(device)

tab_url, tab_upload = st.tabs(["Paste YouTube link", "Upload image"])
img = None
with tab_url:
    url = st.text_input("YouTube URL or video ID",
                        placeholder="https://www.youtube.com/watch?v=...")
    if url:
        vid = extract_video_id(url)
        if not vid:
            st.warning("Couldn't find a video ID in that URL.")
        else:
            img = fetch_thumbnail(vid)
            if img is None:
                st.warning("Couldn't fetch that thumbnail.")
with tab_upload:
    up = st.file_uploader("Thumbnail image", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up).convert("RGB")

if img is not None:
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = float(model(x).softmax(1)[0, 1])
    label, color = verdict(score)

    st.markdown(f"### {color}[{label}]")
    st.progress(score, text=f"Clickbait score: {score:.0%}")

    # Grad-CAM heatmap (needs grads, so a fresh forward/backward outside no_grad)
    cam = GradCAM(model, model.features[-1]).heatmap(x.clone().requires_grad_(True), 1)
    blended = overlay_heatmap(img, cam)
    c1, c2 = st.columns(2)
    c1.image(img, caption="Thumbnail", use_container_width=True)
    c2.image(blended, caption="Where the model looked (red = most)", use_container_width=True)

    st.divider()
    st.caption(
        "Educational project. Trained with **weak (channel-level) labels**, so it "
        "learns thumbnail *style*, not ground truth. Not a verdict on the creator."
    )

with st.sidebar:
    st.header("Model")
    if metrics:
        st.metric("Test accuracy", f"{metrics.get('test_accuracy', 0):.0%}",
                  help=f"ROC-AUC {metrics.get('test_roc_auc')}")
        st.caption(f"Trained on {metrics.get('n_total', '?')} thumbnails "
                   f"(MobileNetV2, transfer learning).")
    st.markdown(
        "**How it works**\n\n"
        "1. A CNN pretrained on millions of images extracts visual features.\n"
        "2. A small trained head scores clickbait vs clean.\n"
        "3. Grad-CAM highlights the pixels that drove the score."
    )
