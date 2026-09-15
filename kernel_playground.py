import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

DISPLAY_SIZE = 360
PREVIEW_MAX_DIM = 64
UPSCALE_FACTOR = 6
BOX_COLOR = (255, 40, 40)

PRESETS = {
    3: {
        "Identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        "Box Blur": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "Gaussian Blur": [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        "Sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "Edge Detect": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "Emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        "Sobel X": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "Sobel Y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    },
    5: {
        "Identity": [[1 if (r, c) == (2, 2) else 0 for c in range(5)] for r in range(5)],
        "Box Blur": [[1] * 5 for _ in range(5)],
        "Gaussian Blur": [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
    },
}

CHANNEL_LABELS = {1: ["Kernel"], 3: ["Kernel (R)", "Kernel (G)", "Kernel (B)"]}


def identity_kernel(k: int):
    center = k // 2
    return [[1 if (r, c) == (center, center) else 0 for c in range(k)] for r in range(k)]


def box_kernel(k: int):
    return [[1] * k for _ in range(k)]


def convolve_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(arr.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros(arr.shape, dtype=np.float64)
    h, w = arr.shape
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


class KernelPlaygroundApp:
    def __init__(self, root: tk.Tk, image_path: str | None):
        self.root = root
        self.root.title("Kernel Playground")

        self.src_path: Path | None = None
        self.original_img: Image.Image | None = None
        self.working_img: Image.Image | None = None
        self.channels = 1

        self.preview_img: Image.Image | None = None
        self.preview_arr: np.ndarray | None = None
        self.full_res_result: Image.Image | None = None

        self.orig_tk: ImageTk.PhotoImage | None = None
        self.output_tk: ImageTk.PhotoImage | None = None

        self.animating = False
        self.anim_job = None

        self.entries: list[list[list[tk.Entry]]] = []
        self.preset_vars: list[tk.StringVar] = []
        self.preset_menus: list[tk.OptionMenu] = []
        self.channel_grid_frames: list[tk.Frame] = []

        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=6)

        tk.Button(top, text="Open Image", command=self.open_image).pack(side="left")

        tk.Label(top, text="Mode:").pack(side="left", padx=(15, 2))
        self.mode_var = tk.StringVar(value="Grayscale")
        tk.OptionMenu(top, self.mode_var, "Grayscale", "Color", command=self.on_mode_change).pack(side="left")

        tk.Label(top, text="Kernel Size:").pack(side="left", padx=(15, 2))
        self.size_var = tk.IntVar(value=3)
        tk.OptionMenu(top, self.size_var, 3, 5, 7, 9, command=self.on_size_change).pack(side="left")

        self.normalize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="Normalize (divide by sum)", variable=self.normalize_var).pack(side="left", padx=15)

        self.save_button = tk.Button(top, text="Save Full-Res Result...", command=self.save_result, state="disabled")
        self.save_button.pack(side="right")

        self.grid_frame = tk.Frame(root)
        self.grid_frame.pack(padx=10, pady=6)

        anim_row = tk.Frame(root)
        anim_row.pack(fill="x", padx=10, pady=6)

        self.apply_button = tk.Button(anim_row, text="Apply to Full Image", command=self.apply_full_res, state="disabled")
        self.apply_button.pack(side="left")

        self.play_button = tk.Button(anim_row, text="Play Animation", command=self.start_animation, state="disabled")
        self.play_button.pack(side="left", padx=10)

        self.stop_button = tk.Button(anim_row, text="Stop", command=self.stop_animation, state="disabled")
        self.stop_button.pack(side="left")

        tk.Label(anim_row, text="Speed:").pack(side="left", padx=(20, 2))
        self.speed_var = tk.IntVar(value=90)
        tk.Scale(anim_row, from_=1, to=100, orient="horizontal", variable=self.speed_var, length=150,
                 label="steps/frame").pack(side="left")

        self.status_label = tk.Label(anim_row, text="")
        self.status_label.pack(side="left", padx=15)

        images_row = tk.Frame(root)
        images_row.pack(padx=10, pady=10)

        left_col = tk.Frame(images_row)
        left_col.pack(side="left", padx=10)
        tk.Label(left_col, text="Original (kernel window shown)").pack()
        self.orig_label = tk.Label(left_col)
        self.orig_label.pack()

        right_col = tk.Frame(images_row)
        right_col.pack(side="left", padx=10)
        tk.Label(right_col, text="Output (builds as animation runs)").pack()
        self.output_label = tk.Label(right_col)
        self.output_label.pack()

        self.build_kernel_grid(3)

        if image_path:
            self.load_image(image_path)

    def build_kernel_grid(self, k: int):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.entries = []
        self.preset_vars = []
        self.preset_menus = []

        labels = CHANNEL_LABELS[self.channels]
        preset_names = list(PRESETS.get(k, {}).keys()) or ["Identity", "Box Blur"]

        for ch, label in enumerate(labels):
            col_frame = tk.Frame(self.grid_frame, relief="groove", borderwidth=1, padx=6, pady=6)
            col_frame.grid(row=0, column=ch, padx=6)

            tk.Label(col_frame, text=label, font=("TkDefaultFont", 10, "bold")).pack()

            preset_var = tk.StringVar(value=preset_names[0])
            preset_menu = tk.OptionMenu(col_frame, preset_var, *preset_names,
                                         command=lambda name, c=ch: self.apply_preset(c, name))
            preset_menu.pack(pady=(2, 6))
            self.preset_vars.append(preset_var)
            self.preset_menus.append(preset_menu)

            entry_grid = tk.Frame(col_frame)
            entry_grid.pack()

            kernel = identity_kernel(k)
            channel_entries = []
            for r in range(k):
                row_entries = []
                for c in range(k):
                    var = tk.StringVar(value=str(kernel[r][c]))
                    entry = tk.Entry(entry_grid, width=5, justify="center", textvariable=var)
                    entry.grid(row=r, column=c, padx=2, pady=2)
                    row_entries.append(entry)
                channel_entries.append(row_entries)
            self.entries.append(channel_entries)

    def on_size_change(self, _val):
        self.build_kernel_grid(self.size_var.get())

    def on_mode_change(self, _val):
        self.channels = 3 if self.mode_var.get() == "Color" else 1
        self.build_kernel_grid(self.size_var.get())
        if self.original_img is not None:
            self.refresh_working_image()

    def apply_preset(self, ch: int, name: str):
        k = self.size_var.get()
        table = PRESETS.get(k, {})
        if name in table:
            values = table[name]
        elif name == "Identity":
            values = identity_kernel(k)
        elif name == "Box Blur":
            values = box_kernel(k)
        else:
            return
        for r in range(k):
            for c in range(k):
                self.entries[ch][r][c].delete(0, "end")
                self.entries[ch][r][c].insert(0, str(values[r][c]))
        self.preset_vars[ch].set(name)

    def read_kernels(self) -> list[np.ndarray] | None:
        k = self.size_var.get()
        kernels = []
        for ch in range(self.channels):
            try:
                values = [[float(self.entries[ch][r][c].get()) for c in range(k)] for r in range(k)]
            except ValueError:
                messagebox.showerror("Invalid kernel", f"All cells in {CHANNEL_LABELS[self.channels][ch]} must be numbers.")
                return None
            kernel = np.array(values, dtype=np.float64)
            if self.normalize_var.get():
                total = kernel.sum()
                if abs(total) > 1e-9:
                    kernel = kernel / total
            kernels.append(kernel)
        return kernels

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        try:
            with Image.open(path) as img:
                self.original_img = img.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return

        self.src_path = Path(path)
        self.root.title(f"Kernel Playground - {self.src_path.name}")
        self.refresh_working_image()

    def refresh_working_image(self):
        if self.mode_var.get() == "Color":
            self.working_img = self.original_img
        else:
            self.working_img = self.original_img.convert("L")

        preview = self.working_img.copy()
        preview.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM))
        self.preview_img = preview
        self.preview_arr = np.array(preview, dtype=np.uint8)

        self.apply_button.config(state="normal")
        self.play_button.config(state="normal")
        self.save_button.config(state="disabled")
        self.full_res_result = None

        blank = np.zeros_like(self.preview_arr)
        self.show_original(None)
        self.show_output(blank)
        self.status_label.config(text="")

    def upscale_for_display(self, arr: np.ndarray) -> Image.Image:
        mode = "L" if arr.ndim == 2 else "RGB"
        img = Image.fromarray(arr.astype(np.uint8), mode=mode)
        w, h = img.size
        scale = max(1, min(DISPLAY_SIZE // max(w, 1), UPSCALE_FACTOR))
        return img.resize((w * scale, h * scale), Image.NEAREST)

    def show_original(self, box: tuple[int, int, int, int] | None):
        base = self.upscale_for_display(self.preview_arr).convert("RGB")
        if box is not None:
            draw = ImageDraw.Draw(base)
            draw.rectangle(box, outline=BOX_COLOR, width=2)
        self.orig_tk = ImageTk.PhotoImage(base)
        self.orig_label.config(image=self.orig_tk)

    def show_output(self, arr: np.ndarray):
        img = self.upscale_for_display(arr)
        self.output_tk = ImageTk.PhotoImage(img)
        self.output_label.config(image=self.output_tk)

    def start_animation(self):
        if self.preview_arr is None or self.animating:
            return
        kernels = self.read_kernels()
        if kernels is None:
            return

        self.animating = True
        self.play_button.config(state="disabled")
        self.stop_button.config(state="normal")

        kh, kw = kernels[0].shape
        pad_h, pad_w = kh // 2, kw // 2

        if self.channels == 1:
            h, w = self.preview_arr.shape
            padded_channels = [np.pad(self.preview_arr.astype(np.float64),
                                       ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")]
            output = np.zeros((h, w), dtype=np.uint8)
        else:
            h, w, _ = self.preview_arr.shape
            padded_channels = [
                np.pad(self.preview_arr[:, :, c].astype(np.float64),
                       ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
                for c in range(3)
            ]
            output = np.zeros((h, w, 3), dtype=np.uint8)

        scale = max(1, min(DISPLAY_SIZE // max(w, 1), UPSCALE_FACTOR))
        positions = [(r, c) for r in range(h) for c in range(w)]
        total = len(positions)
        state = {"idx": 0}

        def step():
            if not self.animating:
                return
            steps_per_frame = max(1, total // max(self.speed_var.get(), 1))
            idx = state["idx"]
            end = min(idx + steps_per_frame, total)
            for i in range(idx, end):
                r, c = positions[i]
                for ch in range(self.channels):
                    window = padded_channels[ch][r:r + kh, c:c + kw]
                    value = float((window * kernels[ch]).sum())
                    value = np.clip(value, 0, 255)
                    if self.channels == 1:
                        output[r, c] = value
                    else:
                        output[r, c, ch] = value
            state["idx"] = end

            r, c = positions[end - 1]
            box = (c * scale, r * scale, (c + kw) * scale, (r + kh) * scale)
            self.show_original(box)
            self.show_output(output)
            self.status_label.config(text=f"{end}/{total} pixels")

            if end < total:
                self.anim_job = self.root.after(20, step)
            else:
                self.animating = False
                self.play_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self.status_label.config(text="Animation complete (preview resolution only)")

        step()

    def stop_animation(self):
        self.animating = False
        if self.anim_job is not None:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self.play_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def apply_full_res(self):
        if self.working_img is None:
            return
        kernels = self.read_kernels()
        if kernels is None:
            return

        arr = np.array(self.working_img, dtype=np.uint8)
        if self.channels == 1:
            result = convolve_same(arr, kernels[0])
            result = np.clip(result, 0, 255).astype(np.uint8)
            self.full_res_result = Image.fromarray(result, mode="L")
        else:
            channel_results = []
            for ch in range(3):
                result = convolve_same(arr[:, :, ch], kernels[ch])
                channel_results.append(np.clip(result, 0, 255).astype(np.uint8))
            stacked = np.stack(channel_results, axis=-1)
            self.full_res_result = Image.fromarray(stacked, mode="RGB")

        preview_result = self.full_res_result.copy()
        preview_result.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM))
        self.show_output(np.array(preview_result, dtype=np.uint8))
        self.show_original(None)
        self.save_button.config(state="normal")
        self.status_label.config(text="Full-resolution result ready to save")

    def save_result(self):
        if self.full_res_result is None or self.src_path is None:
            return
        mode_tag = "color" if self.channels == 3 else "gray"
        default_name = f"{self.src_path.stem}_kernel{self.size_var.get()}_{mode_tag}{self.src_path.suffix}"
        path = filedialog.asksaveasfilename(
            title="Save convolved image",
            initialfile=default_name,
            defaultextension=self.src_path.suffix,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.full_res_result.save(path)
            messagebox.showinfo("Saved", f"Saved to {path}")


if __name__ == "__main__":
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None

    root = tk.Tk()
    app = KernelPlaygroundApp(root, initial_path)
    root.mainloop()
