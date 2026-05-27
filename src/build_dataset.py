import argparse
import os
import subprocess
import sys

import pandas as pd
import requests
from PIL import Image

THUMB_URL = "https://img.youtube.com/vi/{vid}/hqdefault.jpg"

# Curated channels used as WEAK labels for thumbnail style. We assume channels
# known for sensational thumbnails (shocked faces, arrows, ALL-CAPS overlays) lean
# "clickbait", and that news / education / official channels lean "not clickbait".
# Not every thumbnail fits its channel's label -- that noise is the known limitation
# we report honestly; the hand-labeled Kaggle set is what we'd trust for final eval.
CLICKBAIT_CHANNELS = [
    "MrBeast", "SSSniperWolf", "DharMann", "Azzyland", "5MinuteCraftsYouTube",
    "BRIGHTSIDEofficial", "ZHC", "PrestonPlays", "Unspeakable", "CarterSharer",
    "LucasandMarcus", "stokestwins",
]
CLEAN_CHANNELS = [
    "kurzgesagt", "veritasium", "3blue1brown", "TED", "NASA", "BBCNews",
    "AssociatedPress", "CNBC", "mitocw", "khanacademy", "Computerphile",
    "TheRoyalInstitution",
]


def list_videos(handle, limit):
    """Return [(video_id, title), ...] for a channel's recent uploads."""
    url = f"https://www.youtube.com/@{handle}/videos"
    cmd = [
        sys.executable, "-m", "yt_dlp", "--flat-playlist",
        "--playlist-end", str(limit),
        "--print", "%(id)s\t%(title)s", url,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"    [{handle}] timed out")
        return []
    rows = []
    for line in out.stdout.splitlines():
        if "\t" in line:
            vid, title = line.split("\t", 1)
            if vid:
                rows.append((vid.strip(), title.strip()))
    if not rows:
        print(f"    [{handle}] no videos ({out.stderr.strip()[:80]})")
    return rows


def download_thumb(vid, dest, session):
    """Download one thumbnail; return True if it's a real (non-placeholder) image."""
    try:
        r = session.get(THUMB_URL.format(vid=vid), timeout=15)
        if r.status_code != 200 or len(r.content) < 2000:
            return False  # tiny response = missing/placeholder
        with open(dest, "wb") as f:
            f.write(r.content)
        Image.open(dest).verify()  # corrupt check
        return True
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return False


def collect(channels, label, label_name, per_channel, out_root, session):
    rows = []
    cls_dir = os.path.join(out_root, label_name)
    os.makedirs(cls_dir, exist_ok=True)
    for handle in channels:
        vids = list_videos(handle, per_channel)
        got = 0
        for vid, title in vids:
            dest = os.path.join(cls_dir, f"{vid}.jpg")
            if os.path.exists(dest) or download_thumb(vid, dest, session):
                rows.append({"id": vid, "title": title, "channel": handle,
                             "label": label, "label_name": label_name,
                             "path": dest})
                got += 1
        print(f"    [{handle}] {got}/{len(vids)} thumbnails")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-channel", type=int, default=120)
    ap.add_argument("--out", default="data/thumbnails")
    ap.add_argument("--manifest", default="data/manifest.csv")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (clickbait-detector school project)"

    print(f"Collecting CLICKBAIT channels ({len(CLICKBAIT_CHANNELS)})...")
    rows = collect(CLICKBAIT_CHANNELS, 1, "clickbait", args.per_channel, args.out, session)
    print(f"Collecting CLEAN channels ({len(CLEAN_CHANNELS)})...")
    rows += collect(CLEAN_CHANNELS, 0, "clean", args.per_channel, args.out, session)

    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    df.to_csv(args.manifest, index=False)

    n = len(df)
    cb = int(df["label"].sum())
    print(f"\nSaved {n} thumbnails -> {args.manifest}")
    print(f"  clickbait: {cb} ({cb/n:.0%})   clean: {n-cb} ({(n-cb)/n:.0%})")
    print(f"  unique channels: {df['channel'].nunique()}")


if __name__ == "__main__":
    main()
