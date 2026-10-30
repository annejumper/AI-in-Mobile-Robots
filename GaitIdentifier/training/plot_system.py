"""Draw a flowchart of where each part of the project runs.

    .venv/bin/python training/plot_system.py

Writes training/results/system_flowchart.png: laptop vs board (microcontroller
and Linux) vs phone, with the App Lab app outlined.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
BORDER = "#9a9893"
ARROW = "#6f6d68"
APP_LAB = "#2a78d6"
APP_LAB_TINT = "#eaf2fc"
LANE = "#ffffff"


def region(ax, x0, y0, x1, y1, title, subtitle=None, color=BORDER, fill=SURFACE, dashed=False, lw=1.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fill, ec=color, lw=lw, ls=(0, (5, 3)) if dashed else "-"))
    ax.text(x0 + 0.2, y1 - 0.22, title, ha="left", va="top", fontsize=12, weight="bold", color=INK)
    if subtitle:
        ax.text(x0 + 0.2, y1 - 0.52, subtitle, ha="left", va="top", fontsize=9, color=INK_2)


def node(ax, x, y, text, w=3.1, h=0.95, color=BORDER, fill="white", size=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fill, ec=color, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.35)


def arrow(ax, start, end, label=None, both=False, label_dy=0.17, style="arc3"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=ARROW, lw=1.5,
                                shrinkA=0, shrinkB=0, connectionstyle=style))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + label_dy, label, ha="center",
                va="bottom", fontsize=8.5, color=INK_2,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))


def main():
    fig, ax = plt.subplots(figsize=(17, 10.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 17)
    ax.set_ylim(-0.6, 9.8)
    ax.axis("off")

    ax.text(0.2, 9.65, "Anne-kle Monitor: what runs where", fontsize=17, weight="bold", color=INK, va="top")
    ax.text(0.2, 9.15, "Training happens once on the laptop. After deploying, the board does everything "
            "on battery; the phone only displays the board's web page.", fontsize=10.5, color=INK_2, va="top")

    # Laptop: not App Lab
    region(ax, 0.2, 0.3, 4.75, 8.7, "Your laptop (VS Code)", "Not App Lab: plain Python + git")
    lx = 2.65
    node(ax, lx, 7.35, "deploy.py\ncopies the app folder to the board\nover Wi-Fi (SSH)", h=1.0)
    node(ax, lx, 5.85, "Copy recordings off the board\ndata/session_*.csv")
    node(ax, lx, 4.3, "training/train.py\n2 s windows → features (features.py)\n"
         "→ Model 1, Model 2 (+ KNN check)", h=1.1)
    node(ax, lx, 2.75, "model.json (both trees)\n+ plots, tree diagrams")
    node(ax, lx, 1.25, "Edit code: sketch.ino, main.py,\nfeatures.py, index.html")
    arrow(ax, (lx, 5.85 - 0.475), (lx, 4.3 + 0.55))
    arrow(ax, (lx, 4.3 - 0.55), (lx, 2.75 + 0.475))
    # model.json and edited code both go out with the next deploy
    rail = 0.6
    for y in (2.75, 1.25):
        ax.plot([lx - 1.55, rail], [y, y], color=ARROW, lw=1.5)
    ax.plot([rail, rail], [1.25, 7.35], color=ARROW, lw=1.5)
    arrow(ax, (rail, 7.35), (lx - 1.55, 7.35))
    ax.text(rail - 0.12, 4.3, "goes to board on next deploy", rotation=90, ha="right", va="center",
            fontsize=8.5, color=INK_2)

    # Board
    region(ax, 5.1, 0.3, 12.3, 8.7, "Arduino UNO Q (on your ankle, battery powered)")
    node(ax, 6.95, 7.6, "arduino-app-cli  (App Lab's tool)\nbuilds & starts the app; set as default\n"
         "app so it auto-starts on power-up", w=3.4, h=1.0, color=APP_LAB)
    arrow(ax, (lx + 1.55, 7.45), (6.95 - 1.7, 7.6))

    # The App Lab app
    region(ax, 5.3, 0.5, 12.1, 6.85, "App Lab app  (gait-identifier/)", color=APP_LAB, fill=APP_LAB_TINT,
           dashed=True, lw=2)
    arrow(ax, (6.95, 7.1), (6.95, 6.85))
    for x0, x1, title, sub in ((5.5, 8.5, "Linux processor", "python/main.py + WebUI brick"),
                               (9.0, 11.9, "Microcontroller (STM32)", "sketch/sketch.ino")):
        ax.add_patch(FancyBboxPatch((x0, 0.7), x1 - x0, 5.55, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=LANE, ec=APP_LAB, lw=1, alpha=0.9))
        ax.text((x0 + x1) / 2, 6.05, title, ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
        ax.text((x0 + x1) / 2, 5.72, sub, ha="center", va="top", fontsize=8.5, color=INK_2)

    linx, mcux = 7.0, 10.45
    node(ax, linx, 4.75, "Fetch samples, convert to g, °/s\nsave data/session_*.csv", w=2.8)
    node(ax, linx, 3.1, "Every 0.5 s: features of last 2 s\n→ both trees (model.json)\n→ majority of last 3",
         w=2.8, h=1.1)
    node(ax, linx, 1.45, "Web server, port 7000\nserves assets/index.html\n+ /api/status, /api/stream",
         w=2.8, h=1.05)
    node(ax, mcux, 4.75, "Read IMU 100×/s over I²C\nring buffer, tag with label", w=2.6)
    node(ax, mcux, 3.1, "Draw icon on LED matrix\n(prediction or record label)", w=2.6)
    arrow(ax, (linx, 4.75 - 0.475), (linx, 3.1 + 0.55))
    arrow(ax, (linx, 3.1 - 0.55), (linx, 1.45 + 0.525))
    arrow(ax, (mcux - 1.3, 4.75), (linx + 1.4, 4.75), label="Bridge: get_samples", label_dy=0.55)
    arrow(ax, (linx + 1.4, 3.1), (mcux - 1.3, 3.1), label="Bridge: set_matrix,\nset_label", label_dy=0.6)
    arrow(ax, (linx - 1.4, 4.75), (lx + 1.55, 5.85), label="scp when on Wi-Fi", label_dy=0.05)

    # Hardware
    region(ax, 12.65, 3.5, 16.8, 6.25, "Hardware", "on the board / ankle mount")
    node(ax, 14.7, 4.95, "MPU-6050 IMU\n3-axis accel + 3-axis gyro", w=3.2, h=0.85)
    node(ax, 14.7, 4.05 - 0.05, "8×13 LED matrix", w=3.2, h=0.5)
    arrow(ax, (14.7 - 1.6, 4.95), (mcux + 1.3, 4.75))
    arrow(ax, (mcux + 1.3, 3.1), (14.7 - 1.6, 3.95))

    # Phone: not App Lab
    region(ax, 12.65, 0.3, 16.8, 3.15, "Your phone (any browser)", "Not App Lab: just displays the page")
    node(ax, 14.7, 1.35, "Anne-kle Monitor page\nhttp://10.247.137.125:7000\nlive answer, plots, model switch",
         w=3.4, h=1.05)
    arrow(ax, (linx + 1.4, 1.45), (14.7 - 1.7, 1.35), label="same Wi-Fi network: page, data, taps", both=True, label_dy=0.05)

    # Legend
    ax.add_patch(FancyBboxPatch((0.3, -0.42), 0.5, 0.3, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=APP_LAB_TINT, ec=APP_LAB, lw=1.6, ls=(0, (5, 3))))
    ax.text(0.95, -0.27, "Part of the App Lab app (deployed to and runs on the board)", va="center",
            fontsize=9.5, color=INK)
    ax.add_patch(FancyBboxPatch((7.2, -0.42), 0.5, 0.3, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=APP_LAB, lw=1.4))
    ax.text(7.85, -0.27, "App Lab tool (not your app)", va="center", fontsize=9.5, color=INK)
    ax.add_patch(FancyBboxPatch((11.0, -0.42), 0.5, 0.3, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec=BORDER, lw=1.4))
    ax.text(11.65, -0.27, "Not App Lab", va="center", fontsize=9.5, color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "system_flowchart.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
