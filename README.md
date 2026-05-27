# Clickbait Thumbnail Detector

A computer-vision web app that judges YouTube **thumbnails**. Paste a YouTube link
(or upload an image) and a neural network scores how *clickbait* the thumbnail looks,
then shows a **heatmap of where it looked** — the shocked faces, arrows, and big text
that signal clickbait.

**Result:** ~92% accuracy, 0.97 ROC-AUC on held-out thumbnails.

## The problem
Clickbait thumbnails manipulate attention and erode trust. A tool that flags
manipulative thumbnail design supports media literacy — and shows what these
visual tricks have in common.

## Why AI?
"Does this thumbnail *look* like clickbait?" has no fixed rulebook — it's a visual
style learned from thousands of examples. A convolutional neural network learns
these patterns far better than hand-written rules. We use **transfer learning**:
take a CNN (MobileNetV2) already trained on millions of images, freeze it, and train
a small new head to recognize clickbait style.

## How it works
1. **`build_dataset.py`** — uses `yt-dlp` (no API key) to list videos from curated
   channels, then downloads each thumbnail from YouTube's public image URL.
2. **`train_cnn.py`** — transfer learning: MobileNetV2 backbone (frozen) + a fresh
   2-class head, trained with data augmentation. Reports accuracy/precision/recall.
3. **`gradcam.py`** — Grad-CAM produces the "where it looked" heatmap.
4. **`app.py`** — a Streamlit web app for the demo.

## How the data is labeled (and an honest limitation)
There is no large gold-standard clickbait-thumbnail dataset (even researchers note
this). So we use **weak supervision**: thumbnails from channels known for sensational
style are labeled `clickbait`, and those from news/education/official channels are
labeled `clean`. This is a real, citable technique — but it means the model learns
thumbnail *style*, not ground-truth clickbait. The Grad-CAM heatmaps help check that
it focuses on meaningful regions (faces, text) rather than channel logos.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/build_dataset.py          # download thumbnails (~2,800 images)
python src/train_cnn.py --epochs 8   # train + evaluate (uses Apple Silicon GPU if present)
streamlit run src/app.py             # launch the web app
```

Quick test with fewer images:
```bash
python src/build_dataset.py --per-channel 40
```

## Project structure
```
src/
  channels.py        # curated clickbait vs clean channel lists (the weak labels)
  build_dataset.py   # yt-dlp -> video IDs -> download thumbnails -> manifest.csv
  dataset.py         # PyTorch Dataset, train/val/test splits, image transforms
  train_cnn.py       # transfer learning (MobileNetV2), evaluation, saves model
  gradcam.py         # Grad-CAM heatmap utility
  app.py             # Streamlit web app
```

## Citations
- **Thumbnails / video IDs:** collected with [yt-dlp](https://github.com/yt-dlp/yt-dlp);
  images from YouTube's public `img.youtube.com` endpoint.
- **Pretrained model:** MobileNetV2 (ImageNet) via
  [torchvision](https://pytorch.org/vision/stable/models.html).
- **Grad-CAM:** Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization"*, ICCV 2017.
- **Reference dataset (weak-supervision technique):** Zannettou et al., *"The Good,
  the Bad and the Bait,"* dataset DOI 10.5281/zenodo.2546908; and the Kaggle dataset
  `thelazyaz/youtube-clickbait-classification` (CC0).
- **Libraries:** PyTorch, torchvision, scikit-learn, pandas, Pillow, matplotlib,
  Streamlit, requests.
- **AI assistance:** Built with help from Claude (Anthropic) for code scaffolding;
  all code reviewed and modified by me. See presentation for what was AI-assisted.

## Disclaimer
Educational project. Because labels are channel-level (weak), the model reflects
thumbnail *style*, not a verdict on any creator. Not affiliated with YouTube.
