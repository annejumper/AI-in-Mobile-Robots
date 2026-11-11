"""
Train the finger-counting CNN on data/<fingers>/*.jpg.

    python lock/train.py            (or --epochs 40 --lr 0.001)

Data augmentation: every time an image is used it is randomly rotated, shifted, zoomed,
mirrored and re-lit, so the network sees "new" pictures each epoch and doesn't memorise.

Splitting: images are grouped by recording session (the part of the filename before "_"),
and whole sessions go to validation. Neighbouring frames from one recording look almost the same,
so splitting them randomly would let the network be graded on pictures it already studied.

Saves models/finger_cnn.pt and, in results/, the loss/accuracy curves and a confusion matrix.
"""
import argparse
import random
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import config
from model import FingerCNN, preprocess

augment = v2.Compose([
    v2.RandomAffine(degrees=20, translate=(0.12, 0.12), scale=(0.85, 1.15)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
])


class Fingers(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        if path not in self.cache:
            self.cache[path] = preprocess(cv2.imread(str(path)))
        x = self.cache[path]
        if self.train:   # augment in [0, 1] space, then back to [-1, 1]
            x = augment(x * 0.5 + 0.5).clamp(0, 1) * 2 - 1
        return x, label


def split(val_frac, seed):
    """Whole sessions go to validation, per class, so train and val pictures aren't near-copies."""
    rng = random.Random(seed)
    train, val = [], []
    for k, c in enumerate(config.CLASSES):
        sessions = defaultdict(list)
        for p in sorted((config.DATA_DIR / str(c)).glob("*.jpg")):
            sessions[p.name.split("_")[0]].append((p, k))
        names = sorted(sessions)
        rng.shuffle(names)
        total, taken = sum(len(v) for v in sessions.values()), 0
        for name in names:
            if len(names) > 1 and taken < val_frac * total and name != names[-1]:
                val += sessions[name]; taken += len(sessions[name])
            else:
                train += sessions[name]
        if len(names) == 1:   # only one session: fall back to the last 20% of its frames
            items = sessions[names[0]] if names else []
            cut = int(len(items) * (1 - val_frac))
            train = [t for t in train if t not in items[cut:]]
            val += items[cut:]
    return train, val


def run_epoch(model, loader, loss_fn, device, opt=None):
    model.train(opt is not None)
    total_loss, correct, n = 0.0, 0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if opt:
                opt.zero_grad()
                loss.backward()     # backpropagation: d(loss)/d(every weight)
                opt.step()          # gradient descent step
            total_loss += loss.item() * len(y)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            n += len(y)
            preds += p.tolist(); labels += y.tolist()
    return total_loss / max(n, 1), correct / max(n, 1), preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_items, val_items = split(args.val, args.seed)
    print(f"{len(train_items)} training images, {len(val_items)} validation images, on {device}")
    for k, c in enumerate(config.CLASSES):
        print(f"  {c} fingers: {sum(l == k for _, l in train_items)} train, {sum(l == k for _, l in val_items)} val")

    train_dl = DataLoader(Fingers(train_items, True), batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(Fingers(val_items, False), batch_size=args.batch)

    model = FingerCNN().to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} weights")
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = defaultdict(list)
    best = -1.0
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tl, ta, _, _ = run_epoch(model, train_dl, loss_fn, device, opt)
        vl, va, vp, vy = run_epoch(model, val_dl, loss_fn, device)
        for k, v in (("train_loss", tl), ("train_acc", ta), ("val_loss", vl), ("val_acc", va)):
            hist[k].append(v)
        mark = ""
        if va >= best:
            best, best_preds = va, (vp, vy)
            torch.save({"state_dict": model.state_dict(), "img_size": config.IMG_SIZE,
                        "classes": config.CLASSES}, config.MODEL_PATH)
            mark = "  saved"
        print(f"epoch {epoch:3d}  train loss {tl:.3f} acc {ta:.1%}   val loss {vl:.3f} acc {va:.1%}{mark}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["train_loss"], label="train"); a1.plot(hist["val_loss"], label="validation")
    a1.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss"); a1.legend()
    a2.plot(hist["train_acc"], label="train"); a2.plot(hist["val_acc"], label="validation")
    a2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy", ylim=(0, 1.02)); a2.legend()
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "training_curves.png", dpi=150)

    n = len(config.CLASSES)
    cm = np.zeros((n, n), int)
    for p, y in zip(*best_preds):
        cm[y, p] += 1
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(n), yticks=range(n), xlabel="predicted fingers", ylabel="true fingers",
           title=f"Validation confusion matrix ({best:.1%})")
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"\nbest validation accuracy {best:.1%}; model in {config.MODEL_PATH}, plots in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
