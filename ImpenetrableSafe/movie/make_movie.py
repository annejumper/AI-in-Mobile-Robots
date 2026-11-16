"""
Explainer movie for the finger-counting CNN, made from the REAL trained network: the filters,
feature maps, pooling and probabilities shown are what the model actually computes for one of
the training pictures.

    python movie/make_movie.py                 -> movie/cnn_explainer.mp4 (silent, ~2 min)
    python movie/make_movie.py --scene 5       -> just one scene, quick to check
    python movie/make_movie.py --stills        -> one PNG per scene in movie/stills/

Style: steady 30 frames/s, elements fade in, scenes dip through the background color,
SF Pro type, Imp-anne-etrable Safe branding on every scene.
Scene lengths are in SCENES at the bottom; voiceover_script.md has the matching narration.
"""
import argparse
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lock"))
import config                                    # noqa: E402
from model import load_model, preprocess         # noqa: E402

W, H = 1280, 720
FPS = 30
BG = (247, 246, 242)
INK, MUTED, LINE = (29, 29, 31), (110, 110, 118), (214, 210, 202)
RED, BLUE, GREEN = (192, 57, 43), (46, 111, 183), (34, 139, 84)
PANEL = (255, 255, 255)
SF = "/System/Library/Fonts/SFNS.ttf"
WEIGHTS = {"regular": 400, "medium": 510, "semibold": 590, "bold": 700, "heavy": 860}


@lru_cache(maxsize=None)
def font(size, weight="regular"):
    try:
        f = ImageFont.truetype(SF, size)
        f.set_variation_by_axes([100, min(96, max(17, size)), 400, WEIGHTS[weight]])
        return f
    except (OSError, ValueError):
        return ImageFont.load_default(size)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ── The network, run on one real picture ───────────────────────────────────

model = load_model(config.MODEL_PATH, "cpu")


def pick_picture():
    """A board picture of 3 that the network is sure about."""
    best, best_p = None, -1
    for f in sorted((config.DATA_DIR / "3").glob("b*.jpg"))[::7]:
        img = cv2.imread(str(f))
        with torch.no_grad():
            p = torch.softmax(model(preprocess(img)[None]), 1)[0, 3].item()
        if p > best_p:
            best, best_p = img, p
    return best


CROP = pick_picture()
CROP_RGB = cv2.cvtColor(CROP, cv2.COLOR_BGR2RGB)
ACT = {}
with torch.no_grad():
    h = preprocess(CROP)[None]
    names = ["conv1", "bn1", "relu1", "pool1", "conv2", "bn2", "relu2", "pool2",
             "conv3", "bn3", "relu3", "pool3", "conv4", "bn4", "relu4"]
    for name, layer in zip(names, model.features):
        h = layer(h)
        ACT[name] = h[0].numpy()
    gap = model.pool(h).flatten(1)
    scores = model.classifier(gap)
    PROBS = torch.softmax(scores, 1)[0].numpy()
GAP = gap[0].numpy()
SCORES = scores[0].numpy()
INPUT_RGB = cv2.cvtColor(cv2.resize(CROP, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
FILTERS1 = model.features[0].weight.detach().numpy()          # (16, 3, 5, 5)
F1 = int(np.argmax([ACT["relu1"][c].std() for c in range(16)]))   # clearest layer-1 map
N_WEIGHTS = sum(p.numel() for p in model.parameters())


def colorize(m, cmap="magma", center=False):
    m = m.astype(np.float32)
    if center:
        a = np.abs(m).max() + 1e-6
        m = (m / a + 1) / 2
    else:
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return (colormaps[cmap](m)[..., :3] * 255).astype(np.uint8)


_ARRAYS = {}


@lru_cache(maxsize=None)
def _big(key, size):
    return Image.fromarray(_ARRAYS[key]).resize((size, size), Image.NEAREST).convert("RGBA")


def big(rgb, size, key):
    """Blow up a small image with hard pixel edges (cached: the same picture is drawn every frame)."""
    _ARRAYS.setdefault(key, rgb)
    return _big(key, size)


def camera_view():
    snaps = sorted((ROOT / "snaps").glob("snap_frame_*.jpg"))
    x, y, s = config.load_roi()
    frame = cv2.imread(str(snaps[-1])) if snaps else np.full((720, 1280, 3), 225, np.uint8)
    frame[y:y + s, x:x + s] = cv2.resize(CROP, (s, s))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (x, y, s)


CAMERA, ROI = camera_view()


# ── Drawing ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def shadow(w, h):
    pad = 24
    im = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([pad, pad + 4, pad + w, pad + h + 4], 10, fill=(0, 0, 0, 38))
    return im.filter(ImageFilter.GaussianBlur(9))


@lru_cache(maxsize=None)
def resized(key, size):
    return _RAW[key].resize(size).convert("RGBA")


_RAW = {"crop": Image.fromarray(CROP_RGB), "camera": Image.fromarray(CAMERA)}


class Frame:
    """One video frame. Drawing goes into layers; f.at(t0) starts a layer that fades in at t0."""

    def __init__(self, t, dur):
        self.t, self.dur = t, dur
        self.layers = []
        self._layer(1.0)

    def _layer(self, alpha):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.layers.append((img, alpha))
        self.img, self.d = img, ImageDraw.Draw(img)

    def at(self, start, fade=0.45):
        """True once t >= start; what's drawn next fades in over `fade` seconds."""
        if self.t < start:
            return False
        self._layer(ease((self.t - start) / fade))
        return True

    def solid(self):
        """Back to drawing at full strength (after an at() block)."""
        self._layer(1.0)

    def p(self, start, length):
        """0 -> 1 progress of an animation that runs from `start` for `length` seconds."""
        return ease((self.t - start) / length)

    def text(self, xy, s, size=28, color=INK, weight="regular", anchor="la"):
        self.d.text(xy, s, font=font(size, weight), fill=color, anchor=anchor)

    def width(self, s, size, weight="regular"):
        return font(size, weight).getlength(s)

    def heading(self, title, sub=None):
        self.text((64, 92), title, 44, INK, "bold")
        if sub:
            self.text((66, 152), sub, 24, MUTED)

    def card(self, im, xy):
        x, y = int(xy[0]), int(xy[1])
        im = im.convert("RGBA")
        self.img.alpha_composite(shadow(im.width, im.height), (x - 24, y - 24))
        self.img.alpha_composite(im, (x, y))
        self.d.rectangle([x, y, x + im.width - 1, y + im.height - 1], outline=LINE, width=1)

    def panel(self, box, fill=PANEL, radius=14):
        x0, y0, x1, y1 = box
        self.img.alpha_composite(shadow(int(x1 - x0), int(y1 - y0)), (int(x0) - 24, int(y0) - 24))
        self.d.rounded_rectangle(box, radius, fill=fill, outline=LINE, width=1)

    def arrow(self, a, b, color=MUTED, width=3, head=13):
        a, b = np.array(a, float), np.array(b, float)
        v = (b - a) / (np.linalg.norm(b - a) + 1e-6)
        n = np.array([-v[1], v[0]])
        self.d.line([tuple(a), tuple(b - v * head * 0.8)], fill=color, width=width)
        self.d.polygon([tuple(b), tuple(b - head * v + head * 0.5 * n), tuple(b - head * v - head * 0.5 * n)], fill=color)

    def pill(self, center, s, color=RED, size=22):
        w = self.width(s, size, "semibold") + 30
        x, y = center
        hh = size * 0.85
        self.d.rounded_rectangle([x - w / 2, y - hh, x + w / 2, y + hh], int(hh), fill=color)
        self.text((x, y), s, size, (255, 255, 255), "semibold", "mm")


def lock_icon(d, x, y, s, color=RED, hole=BG):
    """A padlock, s pixels tall, top-left corner at (x, y)."""
    sw = max(2, int(s * 0.11))
    d.arc([x + s * 0.18, y, x + s * 0.82, y + s * 0.7], 180, 360, fill=color, width=sw)
    for xx in (x + s * 0.18 + sw / 2, x + s * 0.82 - sw / 2):
        d.line([xx, y + s * 0.34, xx, y + s * 0.48], fill=color, width=sw)
    d.rounded_rectangle([x, y + s * 0.44, x + s, y + s], int(s * 0.12), fill=color)
    r = s * 0.09
    d.ellipse([x + s / 2 - r, y + s * 0.62 - r, x + s / 2 + r, y + s * 0.62 + r], fill=hole)
    d.rectangle([x + s / 2 - r * 0.45, y + s * 0.62, x + s / 2 + r * 0.45, y + s * 0.84], fill=hole)


def brand_title(f, cx, y, size):
    """'The Imp-anne-etrable Safe' with 'anne' in red, centered on cx (y = baseline)."""
    parts = [("The Imp-", INK), ("anne", RED), ("-etrable Safe", INK)]
    x = cx - sum(f.width(p, size, "heavy") for p, _ in parts) / 2
    for p, c in parts:
        f.text((x, y), p, size, c, "heavy", "ls")
        x += f.width(p, size, "heavy")


def chrome(img, name, number, progress):
    """Brand line, scene label and progress bar: identical on every scene, never fades."""
    d = ImageDraw.Draw(img)
    if name:
        lock_icon(d, 64, 30, 20, RED, BG)
        x = 94
        for ch in "THE IMP-ANNE-ETRABLE SAFE":
            d.text((x, 32), ch, font=font(15, "semibold"), fill=RED)
            x += font(15, "semibold").getlength(ch) + 1.6
        d.text((W - 64, 32), f"{number:02d}   {name}", font=font(15, "medium"), fill=MUTED, anchor="ra")
    d.rectangle([0, H - 5, W, H], fill=(230, 227, 220))
    d.rectangle([0, H - 5, int(W * progress), H], fill=RED)


# ── Scenes: each draws one frame; f.t = seconds into the scene ─────────────

def s_title(f):
    lock_icon(f.d, W / 2 - 44, 130, 88, RED, BG)
    brand_title(f, W / 2, 335, 84)
    if f.at(0.9):
        f.text((W / 2, 395), "How a neural network learned to count fingers", 32, MUTED, "regular", "mm")
    if f.at(1.9):
        f.card(resized("crop", (170, 170)), (W // 2 - 85, 460))


def s_camera(f):
    f.heading("The camera", "The network only ever looks inside the red square")
    x, y, s = ROI
    sc = 0.55
    f.card(resized("camera", (int(1280 * sc), int(720 * sc))), (64, 200))
    if f.at(1.2):
        f.d.rectangle([64 + x * sc, 200 + y * sc, 64 + (x + s) * sc, 200 + (y + s) * sc], outline=RED, width=4)
    if f.at(2.8):
        f.arrow((64 + (x + s) * sc + 14, 330), (858, 330), RED)
        f.card(resized("crop", (300, 300)), (880, 200))
    if f.at(4.6, 0.8):                      # the same square, as the network gets it
        f.card(big(INPUT_RGB, 300, "input"), (880, 200))
    if f.at(5.2):
        f.text((1030, 528), "64 × 64 pixels × 3 colors", 24, INK, "semibold", "ma")
    if f.at(6.4):
        lead = "64 × 64 × 3 = 12,288 numbers.  "
        f.text((64, 640), lead, 32, RED, "bold")
        f.text((64 + f.width(lead, 32, "bold"), 640), "That is everything the network sees.", 32, INK)


def s_pixels(f):
    f.heading("A picture is just numbers", "Zooming into one fingertip")
    f.card(big(INPUT_RGB, 320, "input"), (64, 210))
    py, px = 30, 29
    f.d.rectangle([64 + px * 5, 210 + py * 5, 64 + (px + 6) * 5, 210 + (py + 6) * 5], outline=RED, width=3)
    if f.at(1.2):
        f.arrow((64 + (px + 6) * 5 + 10, 210 + py * 5), (510, 240), RED)
    f.solid()
    patch = INPUT_RGB[py:py + 6, px:px + 6]
    cell = 70
    shown = int(36 * f.p(1.5, f.dur - 4.0) + 0.999) if f.t >= 1.5 else 0
    for i in range(6):
        for j in range(6):
            if i * 6 + j >= shown:
                continue
            X0, Y0 = 530 + j * cell, 210 + i * cell
            g = int(np.mean(patch[i, j]))
            f.d.rectangle([X0, Y0, X0 + cell - 2, Y0 + cell - 2], fill=tuple(int(v) for v in patch[i, j]))
            f.text((X0 + cell / 2 - 1, Y0 + cell / 2 - 1), str(g), 22, (255, 255, 255) if g < 145 else INK, "semibold", "mm")
    if f.at(f.dur - 2.2):
        f.text((530, 650), "Bright = big number, dark = small number (gray level 0–255)", 22, MUTED)


def s_neuron(f):
    f.heading("One neuron: the building block", "Multiply each input by a weight, add them up, then ReLU")
    ys = [250, 370, 490]
    for i, y in enumerate(ys):
        f.d.ellipse([140, y - 34, 208, y + 34], fill=PANEL, outline=INK, width=3)
        f.text((174, y), f"x{'₁₂₃'[i]}", 28, INK, "medium", "mm")
    if f.at(1.0):
        for i, y in enumerate(ys):
            f.arrow((212, y), (486, 370 + (y - 370) * 0.25), MUTED, 3)
            f.text((330, (y + 370) / 2 - 26), f"w{'₁₂₃'[i]}", 26, BLUE, "semibold", "mm")
    if f.at(1.6):
        f.d.ellipse([490, 300, 630, 440], fill=PANEL, outline=INK, width=3)
        f.text((560, 370), "Σ + b", 32, INK, "semibold", "mm")
    if f.at(3.0):
        f.arrow((634, 370), (716, 370), MUTED)
        f.panel((720, 300, 900, 440))
        f.d.line([740, 410, 810, 410], fill=RED, width=5)
        f.d.line([810, 410, 880, 325], fill=RED, width=5)
        f.text((810, 458), "ReLU", 24, RED, "semibold", "ma")
        f.arrow((904, 370), (986, 370), MUTED)
        f.text((1000, 370), "output", 28, INK, "medium", "lm")
    if f.at(5.0):
        f.text((140, 580), "z = w₁x₁ + w₂x₂ + w₃x₃ + b        a = max(0, z)", 34, INK, "medium")
        f.text((140, 632), f"One neuron has a handful of weights. This network has {N_WEIGHTS:,}.", 24, MUTED)


def s_conv(f):
    f.heading("Convolution: a filter slides over the picture",
              "At every spot: multiply the 5 × 5 patch (all 3 colors) by the filter's weights, add them up")
    size = 384
    f.card(big(INPUT_RGB, size, "input"), (64, 210))
    fmap = big(colorize(ACT["relu1"][F1]), size, "conv_map")
    steps = 32
    k = int(f.p(0.6, f.dur - 2.6) * steps * steps)
    r, c = divmod(min(k, steps * steps - 1), steps)
    cell = size / steps
    blank = Image.new("RGBA", (size, size), (236, 233, 226, 255))
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([0, 0, size, (k // steps) * cell], fill=255)
    md.rectangle([0, (k // steps) * cell, (k % steps) * cell, (k // steps + 1) * cell], fill=255)
    blank.paste(fmap, (0, 0), mask)
    f.card(blank, (832, 210))
    if k < steps * steps:   # the filter window and where its answer lands
        fx, fy = 64 + c * cell, 210 + r * cell
        f.d.rectangle([fx - 12, fy - 12, fx + cell + 12, fy + cell + 12], outline=RED, width=3)
        f.d.line([fx + cell + 12, fy + cell / 2, 832 + c * cell, 210 + r * cell + cell / 2], fill=(192, 57, 43, 90), width=2)
    f.text((256, 610), "Input  64 × 64", 22, INK, "semibold", "ma")
    f.text((1024, 610), "Feature map  64 × 64", 22, INK, "semibold", "ma")
    f.card(big(colorize(FILTERS1[F1].mean(0), "RdBu_r", center=True), 150, "filter"), (565, 395))
    f.text((640, 560), "The filter: 5 × 5 × 3 = 75 weights", 19, INK, "semibold", "ma")
    f.text((640, 586), "learned, not programmed", 17, MUTED, "regular", "ma")
    if f.at(f.dur - 1.8):
        f.text((640, 655), "It lights up where the picture matches the filter: the edges of the fingers", 24, RED, "medium", "ma")


def s_layer1(f):
    f.heading("Layer 1 has 16 filters", "Each looks for its own pattern: edges, bright skin, shadows, background")
    for i in range(16):
        if f.at(0.25 * i, 0.35):
            r, c = divmod(i, 8)
            f.card(big(colorize(ACT["relu1"][i]), 130, f"l1_{i}"), (64 + c * 145, 220 + r * 160))
    if f.at(f.dur - 2.0):
        f.text((64, 590), "16 feature maps: the same picture, redrawn 16 different ways", 30, RED, "semibold")


def s_relu(f):
    f.heading("ReLU: keep positives, zero the negatives", "The same rule as the single neuron, applied to every number in the map")
    pre = ACT["bn1"][F1]
    f.card(big(colorize(pre, "RdBu_r", center=True), 340, "pre"), (100, 210))
    f.text((270, 572), "Before:  red = positive,  blue = negative", 21, INK, "medium", "ma")
    if f.at(1.8):
        f.arrow((470, 380), (800, 380), RED)
        f.pill((635, 336), "max(0, z)")
    if f.at(3.0):
        f.card(big(colorize(np.where(pre > 0, pre, 0), "RdBu_r", center=True), 340, "post"), (840, 210))
        f.text((1010, 572), "After:  every negative is now 0", 21, INK, "medium", "ma")
    if f.at(5.0):
        f.text((640, 640), "Without ReLU, stacking layers would collapse into one big multiply-and-add.", 24, MUTED, "regular", "ma")


def s_pool(f):
    f.heading("Max pooling", "Keep only the biggest number in every 2 × 2 square: half as wide, half as tall")
    grid = np.array([[1, 3, 0, 2], [5, 2, 1, 0], [0, 1, 4, 6], [2, 0, 3, 1]])
    tint = [(255, 226, 153), (182, 224, 212), (255, 196, 190), (196, 212, 244)]
    cell = 78
    for i in range(4):
        for j in range(4):
            q = (i // 2) * 2 + j // 2
            X0, Y0 = 64 + j * cell, 220 + i * cell
            f.d.rectangle([X0, Y0, X0 + cell - 3, Y0 + cell - 3], fill=tint[q])
            f.text((X0 + cell / 2, Y0 + cell / 2), str(grid[i, j]), 34, INK, "medium", "mm")
    f.arrow((400, 376), (500, 376), RED)
    for q in range(4):
        if f.at(1.8 + q * 0.8, 0.35):
            i, j = divmod(q, 2)
            win = grid[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
            a, b = np.unravel_index(win.argmax(), win.shape)
            X0, Y0 = 64 + (j * 2 + b) * cell, 220 + (i * 2 + a) * cell
            f.d.ellipse([X0 + 12, Y0 + 12, X0 + cell - 15, Y0 + cell - 15], outline=RED, width=4)
            X1, Y1 = 520 + j * cell, 298 + i * cell
            f.d.rectangle([X1, Y1, X1 + cell - 3, Y1 + cell - 3], fill=tint[q])
            f.text((X1 + cell / 2, Y1 + cell / 2), str(win.max()), 34, RED, "bold", "mm")
    if f.at(5.8):
        f.card(big(colorize(ACT["relu1"][F1]), 220, "conv_map"), (800, 220))
        f.arrow((1032, 330), (1078, 330), RED)
        f.card(big(colorize(ACT["pool1"][F1]), 110, "pool1"), (1090, 275))
        f.text((910, 460), "64 × 64", 20, INK, "semibold", "ma")
        f.text((1145, 400), "32 × 32", 20, INK, "semibold", "ma")
    if f.at(7.8):
        f.text((64, 590), "Fewer numbers to work with, and a finger that moves a pixel or two", 26, INK)
        f.text((64, 628), "still gives the same answer.", 26, INK)


def s_deeper(f):
    f.heading("Repeat four times", "Convolution → batch norm → ReLU → max pooling, each layer reading the one before")
    stacks = [("Input", None, 1, "64 × 64 × 3"), ("Layer 1", ACT["pool1"], 16, "32 × 32 × 16"),
              ("Layer 2", ACT["pool2"], 32, "16 × 16 × 32"), ("Layer 3", ACT["pool3"], 64, "8 × 8 × 64"),
              ("Layer 4", ACT["relu4"], 128, "8 × 8 × 128")]
    x = 64
    for n, (name, maps, ch, dims) in enumerate(stacks):
        tile = 168 - n * 20
        depth = min(ch, 8)
        if f.at(n * 1.6, 0.5):
            order = np.argsort([-maps[c].std() for c in range(ch)]) if maps is not None else [0]
            for k in range(depth - 1, -1, -1):
                im = big(INPUT_RGB, tile, "input") if maps is None else big(colorize(maps[order[k]]), tile, f"{name}_{k}")
                f.card(im, (x + k * 7, 240 + k * 7))
            f.text((x + tile / 2, 240 + tile + depth * 7 + 22), name, 22, INK, "semibold", "ma")
            f.text((x + tile / 2, 240 + tile + depth * 7 + 52), dims, 18, MUTED, "regular", "ma")
            if n < len(stacks) - 1 and f.at((n + 1) * 1.6 - 0.3, 0.3):
                f.arrow((x + tile + depth * 7 + 10, 320), (x + tile + depth * 7 + 46, 320), RED)
        x += tile + depth * 7 + 56
    if f.at(f.dur - 3.0):
        f.text((64, 620), "Smaller maps, but more of them: edges → finger parts → whole fingers", 28, RED, "semibold")


def s_gap(f):
    f.heading("Global average pooling", "Each of the 128 final maps is averaged into a single number")
    order = np.argsort(-GAP)[:12]
    for i, c in enumerate(order):
        y = 205 + i * 34
        f.card(big(colorize(ACT["relu4"][c]), 28, f"gap_{c}"), (90, y))
        if f.t >= 1.4 + i * 0.22:
            w = GAP[c] / GAP.max() * 300 * f.p(1.4 + i * 0.22, 0.5)
            f.d.rounded_rectangle([140, y + 5, 140 + max(w, 6), y + 23], 4, fill=BLUE)
            if f.t >= 1.9 + i * 0.22:
                f.text((150 + w, y + 14), f"{GAP[c]:.2f}", 17, INK, "medium", "lm")
    f.text((90, 625), "12 of the 128 shown", 18, MUTED)
    if f.at(5.0):
        f.panel((640, 230, 1216, 520))
        f.text((680, 270), "It no longer matters where the hand is", 28, INK, "semibold")
        f.text((680, 312), "in the square, only how much of each", 28, INK, "semibold")
        f.text((680, 354), "pattern the network found.", 28, INK, "semibold")
        f.text((680, 430), "→  128 numbers", 30, RED, "bold")


def s_decide(f):
    f.heading("The decision", "One last layer turns 128 numbers into 6 scores; softmax turns scores into probabilities")
    for i in range(10):
        y = 215 + i * 38
        a = int(255 * min(1, max(0.12, GAP[i] / GAP.max())))
        f.d.ellipse([90, y, 118, y + 28], fill=(46, 111, 183, a), outline=INK, width=2)
    f.text((104, 610), "128", 20, INK, "semibold", "ma")
    if f.at(1.0):
        for i in range(10):
            for o in range(6):
                f.d.line([120, 229 + i * 38, 378, 236 + o * 64], fill=(110, 110, 118, 70), width=1)
        f.text((470, 192), "Scores", 18, MUTED, "semibold", "ma")
        for o in range(6):
            y = 220 + o * 64
            f.d.ellipse([380, y, 412, y + 32], fill=PANEL, outline=INK, width=2)
            f.text((430, y + 16), f"{o}", 22, INK, "semibold", "lm")
            f.text((560, y + 16), f"{SCORES[o]:+.1f}", 22, INK, "medium", "rm")
    if f.at(3.0):
        f.arrow((610, 380), (690, 380), RED)
        f.text((650, 340), "softmax", 20, RED, "semibold", "ma")
        k = f.p(3.0, 1.6)
        x0, y0, w, h = 730, 210, 460, 300
        bw = w / 6
        for i, pv in enumerate(PROBS):
            bh = max(3, pv * h * k)
            c = RED if i == int(PROBS.argmax()) else (190, 190, 196)
            f.d.rounded_rectangle([x0 + i * bw + 12, y0 + h - bh, x0 + (i + 1) * bw - 12, y0 + h], 5, fill=c)
            f.text((x0 + (i + 0.5) * bw, y0 + h + 12), str(i), 22, INK, "semibold", "ma")
        f.d.line([x0, y0 + h, x0 + w, y0 + h], fill=LINE, width=2)
    if f.at(5.4):
        f.pill((960, 600), f"{int(PROBS.argmax())} fingers  ·  {PROBS.max():.0%} sure", RED, 26)


def s_learn(f):
    f.heading("How it learned: gradient descent", "Repeated over 5,420 labeled pictures, 60 times each")
    steps = ["Show a picture", "Network guesses", "Loss: how wrong?", "Backprop: blame", "Nudge every weight"]
    cx, cy, rad = 330, 410, 185
    active = int(f.t * 1.6) % 5
    for i, st in enumerate(steps):
        a = -np.pi / 2 + i * 2 * np.pi / 5
        x, y = cx + rad * np.cos(a), cy + rad * np.sin(a)
        if f.at(0.5 * i, 0.4):
            on = f.t > 2.6 and i == active
            f.d.rounded_rectangle([x - 104, y - 30, x + 104, y + 30], 30, fill=RED if on else PANEL,
                                  outline=RED if on else LINE, width=2)
            f.text((x, y), st, 20, (255, 255, 255) if on else INK, "semibold", "mm")
    f.solid()
    if f.t > 2.4:
        f.text((cx, cy), "repeat", 22, MUTED, "medium", "mm")
    curves = ROOT / "results" / "training_curves_all.png"
    if curves.exists() and f.at(3.0):
        if "curve" not in _RAW:
            im = Image.open(curves).convert("RGB")
            _RAW["curve"] = im.crop((0, 0, im.width // 2, im.height)).resize((540, 396))
        shown = Image.new("RGB", (540, 396), (255, 255, 255))
        k = f.p(3.2, 3.0)
        shown.paste(_RAW["curve"].crop((0, 0, max(1, int(540 * k)), 396)), (0, 0))
        f.card(shown, (660, 205))
        f.text((930, 618), "Training loss going down", 20, MUTED, "medium", "ma")
    if f.at(f.dur - 2.6):
        f.text((64, 660), f"w_new = w − α · ∂L/∂w     applied to every one of the {N_WEIGHTS:,} weights",
               22, RED, "medium")


def s_safe(f):
    f.heading("The safe", "Press the button, show each digit, press to enter it")
    DIG = {1: [".#.", "##.", ".#.", ".#.", "###"], 4: ["#.#", "#.#", "###", "..#", "..#"],
           5: ["###", "#..", "###", "..#", "###"]}
    CHECK = [".............", "..........#..", ".........##..", "........##...", "..#....##....",
             "..##..##.....", "...####......", "....##......."]
    code = [4, 1, 5]
    x0, y0, p = 110, 225, 34
    f.d.rounded_rectangle([x0 - 24, y0 - 24, x0 + 13 * p + 24, y0 + 8 * p + 24], 16, fill=(24, 24, 28))
    n = sum(f.t > i * 1.8 + 0.4 for i in (1, 2, 3))
    lit = np.zeros((8, 13), bool)
    if f.t > 7.5:
        lit = np.array([[ch == "#" for ch in row] for row in CHECK])
    else:
        for i in range(n):
            for r, row in enumerate(DIG[code[i]]):
                for c, ch in enumerate(row):
                    lit[r + 1, i * 5 + c] |= ch == "#"
    for r in range(8):
        for c in range(13):
            col = (120, 178, 255) if lit[r, c] else (52, 52, 60)
            f.d.ellipse([x0 + c * p + 7, y0 + r * p + 7, x0 + (c + 1) * p - 7, y0 + (r + 1) * p - 7], fill=col)
    f.text((x0 + 13 * p / 2, y0 + 8 * p + 52), "UNO Q LED matrix", 18, MUTED, "medium", "ma")
    lines = [("Press the button", INK), ("Show 4, press", INK), ("Show 1, press", INK), ("Show 5, press", INK),
             ("Code matches: ✓ and the servo opens", GREEN)]
    for i, (ln, col) in enumerate(lines):
        if f.at(i * 1.8, 0.4):
            f.d.ellipse([700, 236 + i * 64, 726, 262 + i * 64], fill=col)
            f.text((713, 249 + i * 64), str(i + 1) if i < 4 else "✓", 15, (255, 255, 255), "bold", "mm")
            f.text((744, 249 + i * 64), ln, 28, col, "semibold" if i == 4 else "regular", "lm")
    if f.at(8.0):
        f.text((744, 590), "Wrong code: ✗ and it stays locked", 24, RED, "medium", "lm")


def s_end(f):
    lock_icon(f.d, W / 2 - 36, 170, 72, RED, BG)
    brand_title(f, W / 2, 345, 76)
    if f.at(0.6):
        f.text((W / 2, 400), "A CNN built from scratch in PyTorch, running on an Arduino UNO Q", 28, INK, "regular", "mm")
        f.text((W / 2, 448), f"4 conv layers  ·  max pooling  ·  global average pooling  ·  {N_WEIGHTS:,} weights",
               20, MUTED, "medium", "mm")


SCENES = [  # (function, seconds, label in the corner)
    (s_title, 5, None), (s_camera, 9, "The camera"), (s_pixels, 8, "Pixels"), (s_neuron, 9, "One neuron"),
    (s_conv, 12, "Convolution"), (s_layer1, 6, "Layer 1"), (s_relu, 8, "ReLU"), (s_pool, 11, "Max pooling"),
    (s_deeper, 11, "Going deeper"), (s_gap, 9, "Global average pooling"), (s_decide, 9, "The decision"),
    (s_learn, 11, "Training"), (s_safe, 10, "The safe"), (s_end, 5, None),
]
TOTAL = sum(d for _, d, _ in SCENES)
FADE = 0.35        # each scene fades in from / out to the background


def render(i, t):
    fn, d, label = SCENES[i]
    fr = Frame(t, d)
    fn(fr)
    scene_alpha = min(ease(t / FADE) if i > 0 else 1.0, ease((d - t) / FADE) if i < len(SCENES) - 1 else 1.0)
    out = Image.new("RGBA", (W, H), BG + (255,))
    for img, a in fr.layers:
        a *= scene_alpha
        if a <= 0.003 or img.getbbox() is None:
            continue
        if a < 0.997:
            img.putalpha(img.getchannel("A").point(lambda v, a=a: int(v * a)))
        out.alpha_composite(img)
    start = sum(dd for _, dd, _ in SCENES[:i])
    chrome(out, label, i, (start + t) / TOTAL)
    return out.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, help="render only scene N (1 = title)")
    ap.add_argument("--stills", action="store_true", help="save the end of every scene as PNG")
    args = ap.parse_args()

    idx = [args.scene - 1] if args.scene else list(range(len(SCENES)))
    if args.stills:
        out = HERE / "stills"
        out.mkdir(exist_ok=True)
        for old in out.glob("scene_*.png"):
            old.unlink()
        for i in idx:
            fn, d, _ = SCENES[i]
            render(i, d - FADE - 0.05).save(out / f"scene_{i + 1:02d}_{fn.__name__[2:]}.png")
        print(f"stills in {out}")
        return

    path = HERE / ("cnn_explainer.mp4" if not args.scene else f"scene_{args.scene:02d}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                          stdin=subprocess.PIPE)
    for i in idx:
        fn, d, _ = SCENES[i]
        for k in range(int(d * FPS)):
            ff.stdin.write(np.asarray(render(i, k / FPS)).tobytes())
        start = sum(dd for _, dd, _ in SCENES[:i])
        print(f"scene {i + 1:2d} {fn.__name__[2:]:8s} {d:3d} s  (starts at {start // 60}:{start % 60:02d})")
    ff.stdin.close()
    ff.wait()
    print(f"\n{path}  ({sum(SCENES[i][1] for i in idx)} s, silent: add your voice-over)")


if __name__ == "__main__":
    main()
