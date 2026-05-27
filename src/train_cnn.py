
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, os.path.dirname(__file__))
from dataset import (ThumbnailDataset, eval_tf, load_manifest, make_splits,  # noqa: E402
                     pick_device, train_tf)


def build_model():
    """MobileNetV2 with frozen backbone + a new 2-class head."""
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2),
    )
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        p = out.softmax(1)[:, 1].cpu()
        ps += out.argmax(1).cpu().tolist()
        probs += p.tolist()
        ys += y.tolist()
    return ys, ps, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="models/clickbait_cnn.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    df = load_manifest(args.manifest)
    splits = make_splits(df)
    print(f"Data: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}")

    train_ds = ThumbnailDataset(splits["train"], train_tf)
    val_ds = ThumbnailDataset(splits["val"], eval_tf)
    test_ds = ThumbnailDataset(splits["test"], eval_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=2)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        ys, ps, _ = evaluate(model, val_dl, device)
        val_acc = accuracy_score(ys, ps)
        print(f"epoch {epoch}/{args.epochs}  train_loss={running/len(train_ds):.3f}  "
              f"val_acc={val_acc:.3f}")
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    print("\n=== TEST performance (held-out) ===")
    ys, ps, probs = evaluate(model, test_dl, device)
    acc = accuracy_score(ys, ps)
    auc = roc_auc_score(ys, probs)
    print(f"Accuracy: {acc:.3f}   ROC-AUC: {auc:.3f}")
    print(classification_report(ys, ps, target_names=["clean", "clickbait"]))
    print("Confusion matrix [rows=true clean/clickbait, cols=pred]:")
    print(confusion_matrix(ys, ps))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "mobilenet_v2"}, args.out)
    with open(args.out.replace(".pt", "_metrics.json"), "w") as f:
        json.dump({"test_accuracy": round(acc, 3), "test_roc_auc": round(auc, 3),
                   "best_val_acc": round(best_val, 3), "n_total": len(df)}, f, indent=2)
    print(f"\nSaved model -> {args.out}")


if __name__ == "__main__":
    main()
