"""
Stop-motion style explainer of the finger-counting CNN, made from the REAL trained network:
the filters, feature maps, pooling and probabilities shown are what the model actually computes
for one of the training pictures.

    python movie/make_movie.py                 -> movie/cnn_explainer.mp4 (silent, ~100 s)
    python movie/make_movie.py --scene 5       -> just one scene, quick to check
    python movie/make_movie.py --stills        -> one PNG per scene in movie/stills/

Stop-motion look: 6 drawings a second, and every drawing is nudged and tilted a tiny bit at
random, like paper cut-outs moved by hand between photos.
Scene lengths are in SCENES at the bottom; voiceover_script.md has the matching narration.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lock"))
import config                                    # noqa: E402
from model import load_model, preprocess         # noqa: E402

W, H = 1280, 720
DRAWINGS_PER_S = 6
INK, RED, BLUE, GREEN, GREY = (40, 40, 45), (192, 57, 43), (46, 111, 183), (39, 135, 80), (120, 120, 125)
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
rng = np.random.default_rng(7)


SUBS = {"×": "x", "→": "->", "−": "-", "₁": "1", "₂": "2", "₃": "3", "✓": "check", "✗": "X", "α": "alpha"}


def font(size, title=False):
    path = FONT_DIR / ("MarkerFelt.ttc" if title else "ChalkboardSE.ttc")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default(size)


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
X = preprocess(CROP)[None]
ACT = {}                       # name -> activations (C, H, W) as numpy
with torch.no_grad():
    h = X
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
# the layer-1 filter whose map shows the fingers most clearly (biggest spread of values)
F1 = int(np.argmax([ACT["relu1"][c].std() for c in range(16)]))


def colorize(m, cmap="magma", center=False):
    """2-D array -> RGB uint8 using a matplotlib colour map."""
    m = m.astype(np.float32)
    if center:
        a = np.abs(m).max() + 1e-6
        m = (m / a + 1) / 2
    else:
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return (colormaps[cmap](m)[..., :3] * 255).astype(np.uint8)


def big(rgb, size):
    """Blow up a small image with hard pixel edges."""
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


def frame_with_hand():
    """The full camera view: the empty-box snapshot with the hand crop put back in the red square."""
    snaps = sorted((ROOT / "snaps").glob("snap_frame_*.jpg"))
    roi = config.load_roi()
    if snaps:
        frame = cv2.imread(str(snaps[-1]))
        x, y, s = roi
        frame[y:y + s, x:x + s] = cv2.resize(CROP, (s, s))
    else:
        frame = np.full((720, 1280, 3), 220, np.uint8)
        x, y, s = 320, 40, 640
        frame[y:y + s, x:x + s] = cv2.resize(CROP, (s, s))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), roi


# ── Drawing helpers ────────────────────────────────────────────────────────

def paper():
    base = np.full((H, W, 3), (246, 241, 228), np.float32)
    grain = rng.normal(0, 4, (H // 4, W // 4))
    grain = cv2.resize(grain, (W, H), interpolation=cv2.INTER_CUBIC)[..., None]
    vignette = 1 - 0.12 * (((np.indices((H, W))[1] - W / 2) / W) ** 2 + ((np.indices((H, W))[0] - H / 2) / H) ** 2)[..., None]
    return Image.fromarray(np.clip((base + grain) * vignette, 0, 255).astype(np.uint8))


PAPER = paper()


class Sheet:
    """One drawing: a transparent layer we draw on, later dropped onto the paper with a wobble."""

    def __init__(self):
        self.img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def text(self, xy, s, size=40, color=INK, title=False, anchor="la"):
        for a, b in SUBS.items():          # the handwriting fonts lack these symbols
            s = s.replace(a, b)
        self.d.text(xy, s, font=font(size, title), fill=color, anchor=anchor)

    def card(self, im, xy, border=INK, width=4, shadow=True):
        """Paste a picture like a photo cut-out: drop shadow and an outline."""
        x, y = xy
        if shadow:
            self.d.rectangle([x + 6, y + 6, x + im.width + 6, y + im.height + 6], fill=(0, 0, 0, 45))
        self.img.paste(im.convert("RGBA"), (x, y))
        self.d.rectangle([x, y, x + im.width, y + im.height], outline=border, width=width)

    def arrow(self, a, b, color=INK, width=5):
        self.d.line([a, b], fill=color, width=width)
        v = np.array(b, float) - np.array(a, float)
        v /= np.linalg.norm(v) + 1e-6
        n = np.array([-v[1], v[0]])
        tip = np.array(b, float)
        self.d.polygon([tuple(tip), tuple(tip - 18 * v + 9 * n), tuple(tip - 18 * v - 9 * n)], fill=color)

    def bars(self, values, x, y, w, h, labels=None, color=BLUE, highlight=None, size=26):
        n = len(values)
        bw = w / n
        for i, v in enumerate(values):
            bh = max(2, v * h)
            c = RED if i == highlight else color
            self.d.rectangle([x + i * bw + 6, y + h - bh, x + (i + 1) * bw - 6, y + h], fill=c, outline=INK, width=2)
            if labels:
                self.text((x + (i + 0.5) * bw, y + h + 8), labels[i], size, anchor="ma")
        self.d.line([x, y + h, x + w, y + h], fill=INK, width=3)


def wobble(sheet):
    """Stop-motion: each drawing lands a pixel or two off, slightly tilted."""
    angle = rng.uniform(-0.35, 0.35)
    dx, dy = rng.integers(-3, 4, 2)
    layer = sheet.img.rotate(angle, resample=Image.BICUBIC, translate=(int(dx), int(dy)))
    out = PAPER.copy()
    out.paste(layer, (0, 0), layer)
    return out


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def heading(s, title, sub=None):
    s.text((60, 40), title, 54, INK, title=True)
    if sub:
        s.text((62, 108), sub, 28, GREY)


# ── Scenes: each is f(sheet, t, dur) with t in seconds ─────────────────────

def s_title(s, t, d):
    s.text((W / 2, 230), "The Imp-anne-etrable Safe", 92, RED, title=True, anchor="mm")
    if t > 1.2:
        s.text((W / 2, 340), "how a neural network counts fingers", 44, INK, anchor="mm")
    if t > 2.4:
        im = Image.fromarray(cv2.cvtColor(CROP, cv2.COLOR_BGR2RGB)).resize((200, 200))
        s.card(im, (W // 2 - 100, 410))
        s.text((W / 2 + 130, 520), "?", 120, RED, title=True, anchor="mm")


def s_camera(s, t, d):
    heading(s, "1. The camera", "the network only ever looks inside the red square")
    full, (x, y, sz) = frame_with_hand()
    scale = 0.55
    im = Image.fromarray(full).resize((int(1280 * scale), int(720 * scale)))
    s.card(im, (60, 170))
    if t > 1.5:
        s.d.rectangle([60 + x * scale, 170 + y * scale, 60 + (x + sz) * scale, 170 + (y + sz) * scale],
                      outline=RED, width=6)
    if t > 3:
        s.arrow((60 + (x + sz) * scale + 10, 300), (860, 300), RED)
        k = ease((t - 3) / 2.5)
        size = int(280 - 150 * k)                      # the crop shrinks...
        crop = Image.fromarray(cv2.cvtColor(CROP, cv2.COLOR_BGR2RGB)).resize((64, 64), Image.BILINEAR)
        s.card(crop.resize((260, 260), Image.NEAREST) if k >= 1 else
               Image.fromarray(cv2.cvtColor(CROP, cv2.COLOR_BGR2RGB)).resize((size, size)), (880, 170))
        if k >= 1:
            s.text((1010, 450), "shrunk to 64 × 64", 34, INK, anchor="ma")
            s.text((1010, 495), "× 3 colours (R, G, B)", 30, GREY, anchor="ma")
    if t > 6.5:
        s.text((60, 600), "= 64 × 64 × 3 = 12,288 numbers. That's all the network gets.", 38, RED)


def s_pixels(s, t, d):
    heading(s, "2. A picture is just numbers", "zooming into the corner of one fingertip")
    s.card(big(INPUT_RGB, 320), (60, 180))
    # an 6x6 patch around the middle fingertip
    py, px = 30, 29
    s.d.rectangle([60 + px * 5, 180 + py * 5, 60 + (px + 6) * 5, 180 + (py + 6) * 5], outline=RED, width=3)
    if t > 1.5:
        s.arrow((60 + (px + 6) * 5 + 8, 180 + py * 5), (500, 220), RED)
        patch = INPUT_RGB[py:py + 6, px:px + 6]
        cell = 72
        shown = int(min(36, (t - 1.5) / (d - 3.5) * 36)) if t < d - 2 else 36
        for i in range(6):
            for j in range(6):
                if i * 6 + j >= shown:
                    continue
                X0, Y0 = 520 + j * cell, 180 + i * cell
                rgb = tuple(int(v) for v in patch[i, j])
                s.d.rectangle([X0, Y0, X0 + cell, Y0 + cell], fill=rgb + (255,), outline=INK, width=2)
                g = int(np.mean(patch[i, j]))
                s.text((X0 + cell / 2, Y0 + cell / 2), str(g), 24, (255, 255, 255) if g < 140 else INK, anchor="mm")
    if t > d - 2:
        s.text((520, 640), "bright = big number, dark = small (grey level, 0-255)", 28, GREY)


def s_neuron(s, t, d):
    heading(s, "3. One neuron (like the V-fit)", "multiply, add, then ReLU")
    xs = [(140, 230), (140, 360), (140, 490)]
    labels = ["x₁", "x₂", "x₃"]
    for (x, y), lab, w in zip(xs, labels, ["w₁", "w₂", "w₃"]):
        s.d.ellipse([x - 34, y - 34, x + 34, y + 34], outline=INK, width=4)
        s.text((x, y), lab, 34, anchor="mm")
        if t > 1:
            s.arrow((x + 36, y), (470, 360), GREY, 4)
            s.text(((x + 470) / 2 - 10, (y + 360) / 2 - 34), w, 30, BLUE, anchor="mm")
    if t > 1.5:
        s.d.ellipse([470, 290, 610, 430], outline=INK, width=5, fill=(255, 250, 235, 255))
        s.text((540, 360), "Σ + b", 36, anchor="mm")
    if t > 3:
        s.arrow((612, 360), (700, 360))
        s.d.rectangle([705, 300, 885, 420], outline=INK, width=4)
        s.d.line([725, 400, 795, 400], fill=RED, width=5)
        s.d.line([795, 400, 865, 320], fill=RED, width=5)
        s.text((795, 435), "ReLU", 30, RED, anchor="ma")
        s.arrow((888, 360), (980, 360))
        s.text((1000, 360), "output", 34, anchor="lm")
    if t > 5:
        s.text((140, 580), "z = w₁x₁ + w₂x₂ + w₃x₃ + b        a = max(0, z)", 40)
        s.text((140, 635), "the V-fit had one input (m, b); this network has 99,222 weights", 30, GREY)


def s_conv(s, t, d):
    heading(s, "4. Convolution: a 5×5 filter slides over the picture",
            "at every spot: multiply the 5×5 patch (all 3 colours) by the filter's weights, add them up")
    size = 384
    s.card(big(INPUT_RGB, size), (60, 170))
    fmap = colorize(ACT["relu1"][F1])
    steps = 16                                   # the filter visits a 16 x 16 grid of spots on screen
    k = min(int(t / (d - 1.5) * steps * steps), steps * steps)
    r, c = divmod(min(k, steps * steps - 1), steps)
    cell = size / steps
    # the sliding filter window
    if k < steps * steps:
        fx, fy = 60 + c * cell, 170 + r * cell
        s.d.rectangle([fx - 6, fy - 6, fx + cell + 6, fy + cell + 6], outline=RED, width=5)
    # the output map, revealed up to where the filter has been
    out = big(fmap, size).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    full_rows = k // steps
    md.rectangle([0, 0, size, full_rows * cell], fill=255)
    md.rectangle([0, full_rows * cell, (k % steps) * cell, (full_rows + 1) * cell], fill=255)
    blank = Image.new("RGBA", (size, size), (235, 230, 215, 255))
    blank.paste(out, (0, 0), mask)
    s.card(blank, (836, 170))
    s.arrow((460, 360), (820, 360), RED)
    # the filter itself, magnified
    f = FILTERS1[F1].mean(0)
    s.card(big(colorize(f, "RdBu_r", center=True), 170), (555, 420), width=3)
    s.text((640, 600), "the filter: 5 × 5 × 3 = 75 weights", 24, INK, anchor="ma")
    s.text((640, 632), "(learned; colours averaged)", 22, GREY, anchor="ma")
    s.text((252, 568), "input 64 × 64", 28, anchor="ma")
    s.text((1028, 568), "feature map 64 × 64", 28, anchor="ma")
    if t > d - 1.5:
        s.text((850, 625), "lights up where the picture", 28, RED)
        s.text((850, 662), "matches the filter: finger edges", 28, RED)


def s_layer1(s, t, d):
    heading(s, "Layer 1 has 16 different filters", "each one looks for its own pattern: edges, light, dark, skin colour…")
    maps = ACT["relu1"]
    shown = int(min(16, t / (d - 2) * 16) + 1)
    for i in range(min(shown, 16)):
        r, c = divmod(i, 8)
        s.card(big(colorize(maps[i]), 132), (60 + c * 148, 190 + r * 170), width=3, shadow=False)
    if t > d - 2:
        s.text((60, 560), "16 feature maps: the picture redrawn 16 ways", 38, RED)


def s_relu(s, t, d):
    heading(s, "5. ReLU: keep the positives, zero the negatives", "same rule as the single neuron, applied to every number in the map")
    pre = ACT["bn1"][F1]
    post = ACT["relu1"][F1]
    s.card(big(colorize(pre, "RdBu_r", center=True), 360), (80, 180))
    s.text((260, 560), "before: red +, blue −", 30, anchor="ma")
    if t > 2:
        s.arrow((460, 360), (800, 360), RED)
        s.text((630, 290), "max(0, z)", 40, RED, anchor="ma")
    if t > 3:
        neg_zero = np.where(pre > 0, pre, 0)
        s.card(big(colorize(neg_zero, "RdBu_r", center=True), 360), (840, 180))
        s.text((1020, 560), "after: negatives → 0", 30, anchor="ma")
    if t > 5:
        s.text((80, 625), "without ReLU, stacking layers would just be one big multiply-and-add", 30, GREY)


def s_pool(s, t, d):
    heading(s, "6. Max pooling: keep the biggest in every 2 × 2 square", "the map gets half as wide and half as tall")
    grid = np.array([[1, 3, 0, 2], [5, 2, 1, 0], [0, 1, 4, 6], [2, 0, 3, 1]])
    cols = [(255, 214, 102), (159, 211, 199), (255, 170, 165), (175, 196, 240)]
    cell = 80
    for i in range(4):
        for j in range(4):
            q = (i // 2) * 2 + j // 2
            X0, Y0 = 80 + j * cell, 190 + i * cell
            s.d.rectangle([X0, Y0, X0 + cell, Y0 + cell], fill=cols[q] + (255,), outline=INK, width=3)
            win = grid[(i // 2) * 2:(i // 2) * 2 + 2, (j // 2) * 2:(j // 2) * 2 + 2]
            is_max = grid[i, j] == win.max() and t > 2 + q * 0.8
            s.text((X0 + cell / 2, Y0 + cell / 2), str(grid[i, j]), 40, RED if is_max else INK, anchor="mm")
            if is_max:
                s.d.ellipse([X0 + 14, Y0 + 14, X0 + cell - 14, Y0 + cell - 14], outline=RED, width=4)
    s.arrow((420, 350), (520, 350), RED)
    for q in range(4):
        if t > 2 + q * 0.8:
            i, j = divmod(q, 2)
            X0, Y0 = 540 + j * cell, 270 + i * cell
            m = grid[i * 2:i * 2 + 2, j * 2:j * 2 + 2].max()
            s.d.rectangle([X0, Y0, X0 + cell, Y0 + cell], fill=cols[q] + (255,), outline=INK, width=3)
            s.text((X0 + cell / 2, Y0 + cell / 2), str(m), 40, RED, anchor="mm")
    if t > 6:
        s.card(big(colorize(ACT["relu1"][F1]), 200), (820, 190), width=3)
        s.arrow((1030, 290), (1080, 290), RED)
        s.card(big(colorize(ACT["pool1"][F1]), 100), (1090, 240), width=3)
        s.text((920, 400), "64 × 64  →  32 × 32", 30, anchor="ma")
    if t > 8:
        s.text((80, 560), "fewer numbers to work with, and a finger that moves", 34, INK)
        s.text((80, 605), "a pixel or two still gives the same answer", 34, INK)


def s_deeper(s, t, d):
    heading(s, "7. Repeat: conv → batch norm → ReLU → pool", "each layer looks at the maps from the layer before")
    stacks = [("input", INPUT_RGB[None], 64, 1, "64×64×3"),
              ("layer 1", ACT["pool1"], 32, 16, "32×32×16"),
              ("layer 2", ACT["pool2"], 16, 32, "16×16×32"),
              ("layer 3", ACT["pool3"], 8, 64, "8×8×64"),
              ("layer 4", ACT["relu4"], 8, 128, "8×8×128")]
    x = 60
    for n, (name, maps, res, ch, dims) in enumerate(stacks):
        if t < n * 1.6:
            break
        tile = 170 - n * 20
        depth = min(ch, 8)
        for k in range(depth - 1, -1, -1):
            off = k * 7
            if n == 0:
                im = big(INPUT_RGB, tile)
            else:
                order = np.argsort([-maps[c].std() for c in range(ch)])
                im = big(colorize(maps[order[k]]), tile)
            s.card(im, (x + off, 230 + off), width=2, shadow=False)
        s.text((x + tile / 2, 230 + tile + depth * 7 + 18), name, 30, anchor="ma")
        s.text((x + tile / 2, 230 + tile + depth * 7 + 56), dims, 26, GREY, anchor="ma")
        if n < len(stacks) - 1 and t > (n + 1) * 1.6:
            s.arrow((x + tile + depth * 7 + 8, 300), (x + tile + depth * 7 + 52, 300), RED, 4)
        x += tile + depth * 7 + 56
    if t > d - 3:
        s.text((60, 600), "smaller pictures, more of them: from edges → finger parts → whole fingers", 34, RED)


def s_gap(s, t, d):
    heading(s, "8. Global average pooling", "each of the 128 final maps is averaged into ONE number")
    maps = ACT["relu4"]
    order = np.argsort(-GAP)[:12]
    for i, c in enumerate(order):
        y = 175 + i * 37
        s.card(big(colorize(maps[c]), 32), (90, y), width=2, shadow=False)
        if t > 1.5 + i * 0.25:
            s.arrow((132, y + 16), (210, y + 16), GREY, 3)
            w = GAP[c] / GAP.max() * 300
            s.d.rectangle([220, y + 4, 220 + w, y + 28], fill=BLUE, outline=INK, width=2)
            s.text((230 + w, y + 16), f"{GAP[c]:.2f}", 22, INK, anchor="lm")
    s.text((90, 630), "(12 of the 128 shown)", 24, GREY)
    if t > 5:
        s.text((680, 240), "it no longer matters WHERE", 38, RED)
        s.text((680, 290), "the hand is in the square,", 38, RED)
        s.text((680, 340), "only HOW MUCH of each", 38, RED)
        s.text((680, 390), "pattern the network found.", 38, RED)
        s.text((680, 470), "→ 128 numbers", 40, INK)


def s_decide(s, t, d):
    heading(s, "9. The decision", "one last layer of neurons turns 128 numbers into 6 scores, then softmax → probabilities")
    for i in range(10):
        y = 200 + i * 40
        s.d.ellipse([90, y, 120, y + 30], outline=INK, width=3, fill=(46, 111, 183, int(255 * min(1, GAP[i] / GAP.max()))))
    s.text((105, 610), "128", 30, anchor="ma")
    if t > 1:
        for i in range(10):
            for o in range(6):
                s.d.line([122, 215 + i * 40, 380, 215 + o * 66], fill=(120, 120, 125, 90), width=1)
        for o in range(6):
            y = 200 + o * 66
            s.d.ellipse([380, y, 412, y + 32], outline=INK, width=3, fill=(255, 250, 235, 255))
            s.text((430, y + 16), f"{o}:  {SCORES[o]:+.1f}", 28, anchor="lm")
        s.text((430, 160), "scores", 28, GREY)
    if t > 3:
        s.arrow((700, 380), (770, 380), RED)
        s.text((735, 330), "softmax", 28, RED, anchor="ma")
        k = ease((t - 3) / 2)
        s.bars(PROBS * k, 790, 200, 420, 300, labels=[str(i) for i in range(6)], highlight=int(PROBS.argmax()))
    if t > 5.5:
        s.text((1000, 600), f"{int(PROBS.argmax())} fingers, {PROBS.max():.0%} sure", 44, RED, anchor="ma")


def s_learn(s, t, d):
    heading(s, "10. How it learned: gradient descent, like the V-fit", "5,420 labelled pictures, shown 60 times each")
    steps = ["show a picture", "network guesses", "loss = how wrong", "backprop: blame", "nudge every weight"]
    cx, cy, rad = 330, 405, 200
    for i, st in enumerate(steps):
        a = -np.pi / 2 + i * 2 * np.pi / len(steps)
        x, y = cx + rad * np.cos(a), cy + rad * np.sin(a)
        on = t > 0.6 * i
        s.d.ellipse([x - 110, y - 40, x + 110, y + 40], outline=RED if int(t * 2) % 5 == i else INK, width=4,
                    fill=(255, 250, 235, 255) if on else (0, 0, 0, 0))
        if on:
            s.text((x, y), st, 21, anchor="mm")
    s.text((cx, cy), "repeat", 30, RED, anchor="mm")
    curves = ROOT / "results" / "training_curves_all.png"
    if curves.exists() and t > 3:
        im = Image.open(curves).convert("RGB")
        im = im.crop((0, 0, im.width // 2, im.height)).resize((560, 410))
        k = ease((t - 3) / 3)
        shown = im.crop((0, 0, max(1, int(560 * k)), 410))
        s.card(shown, (640, 175), width=3)
        s.text((920, 600), "loss goes down as it learns", 30, anchor="ma")
    if t > d - 2.5:
        s.text((60, 640), "w_new = w − α · dL/dw   (same update as m and b in the V-fit, for all 99,222 weights)", 26, RED)


def s_safe(s, t, d):
    heading(s, "11. The safe", "button → 3 digits → check")
    code = [4, 1, 5]
    DIG = {1: [".#.", "##.", ".#.", ".#.", "###"], 4: ["#.#", "#.#", "###", "..#", "..#"],
           5: ["###", "#..", "###", "..#", "###"]}
    # LED matrix 13 x 8
    x0, y0, p = 120, 200, 34
    s.d.rectangle([x0 - 20, y0 - 20, x0 + 13 * p + 20, y0 + 8 * p + 20], fill=(25, 25, 30, 255), outline=INK, width=4)
    n = sum(t > i * 1.8 + 0.4 for i in (1, 2, 3))      # each digit appears just after its line
    lit = np.zeros((8, 13), bool)
    if t > 7.5:
        chk = [".............", "..........#..", ".........##..", "........##...", "..#....##....",
               "..##..##.....", "...####......", "....##......."]
        lit = np.array([[ch == "#" for ch in row] for row in chk])
    else:
        for i in range(n):
            for r, row in enumerate(DIG[code[i]]):
                for c, ch in enumerate(row):
                    if ch == "#":
                        lit[r + 1, i * 5 + c] = True
    for r in range(8):
        for c in range(13):
            col = (110, 170, 255, 255) if lit[r, c] else (55, 55, 65, 255)
            s.d.ellipse([x0 + c * p + 6, y0 + r * p + 6, x0 + (c + 1) * p - 6, y0 + (r + 1) * p - 6], fill=col)
    lines = ["press the button", "show 4 → press", "show 1 → press", "show 5 → press", "code matches → ✓, servo opens"]
    for i, ln in enumerate(lines):
        if t > i * 1.8:
            s.text((700, 210 + i * 62), ln, 36, GREEN if i == 4 else INK)
    if t > 7.5:
        s.text((700, 560), "wrong code → ✗, stays locked", 32, RED)


def s_end(s, t, d):
    s.text((W / 2, 280), "The Imp-anne-etrable Safe", 80, RED, title=True, anchor="mm")
    s.text((W / 2, 380), "a CNN built from scratch in PyTorch, running on an Arduino UNO Q", 36, INK, anchor="mm")
    s.text((W / 2, 440), "4 conv layers · max pooling · global average pooling · 99,222 weights", 30, GREY, anchor="mm")


SCENES = [  # (function, seconds)
    (s_title, 5), (s_camera, 9), (s_pixels, 8), (s_neuron, 9), (s_conv, 12), (s_layer1, 6),
    (s_relu, 8), (s_pool, 11), (s_deeper, 11), (s_gap, 9), (s_decide, 9), (s_learn, 11),
    (s_safe, 10), (s_end, 5),
]


def render(fn, t, d):
    s = Sheet()
    fn(s, t, d)
    return wobble(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, help="render only scene N (1 = title)")
    ap.add_argument("--stills", action="store_true", help="save the last drawing of every scene as PNG")
    args = ap.parse_args()

    scenes = list(enumerate(SCENES, 1))
    if args.scene:
        scenes = [scenes[args.scene - 1]]
    if args.stills:
        out = HERE / "stills"
        out.mkdir(exist_ok=True)
        for i, (fn, d) in scenes:
            render(fn, d - 0.01, d).save(out / f"scene_{i:02d}_{fn.__name__[2:]}.png")
        print(f"stills in {out}")
        return

    path = HERE / ("cnn_explainer.mp4" if not args.scene else f"scene_{args.scene:02d}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-framerate", str(DRAWINGS_PER_S), "-i", "-",
                           "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
                          stdin=subprocess.PIPE)
    total = 0
    for i, (fn, d) in scenes:
        n = int(d * DRAWINGS_PER_S)
        for f in range(n):
            ff.stdin.write(np.asarray(render(fn, f / DRAWINGS_PER_S, d).convert("RGB")).tobytes())
        total += d
        print(f"scene {i:2d} {fn.__name__[2:]:8s} {d:3d} s  (starts at {total - d:3d} s)")
    ff.stdin.close()
    ff.wait()
    print(f"\n{path}  ({total} s, silent: add your voice-over)")


if __name__ == "__main__":
    main()
