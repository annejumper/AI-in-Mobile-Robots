import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageChops, ImageFilter, ImageTk

MAX_PREVIEW_SIZE = 700
MAX_MORPH_ITERATIONS = 10


class ThresholdApp:
    def __init__(self, root: tk.Tk, image_path: str | None):
        self.root = root
        self.root.title("Grayscale Threshold")

        self.src_path: Path | None = None
        self.grayscale_img: Image.Image | None = None
        self.preview_img: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=10, pady=8)

        tk.Button(controls, text="Open Image", command=self.open_image).pack(side="left")

        self.threshold_var = tk.IntVar(value=128)
        self.slider = tk.Scale(
            controls,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.threshold_var,
            command=lambda _val: self.update_preview(),
            length=400,
            label="Threshold",
        )
        self.slider.pack(side="left", padx=10, fill="x", expand=True)

        self.erode_var = tk.IntVar(value=0)
        self.erode_slider = tk.Scale(
            controls,
            from_=0,
            to=MAX_MORPH_ITERATIONS,
            orient="horizontal",
            variable=self.erode_var,
            command=lambda _val: self.update_preview(),
            length=150,
            label="Erode",
        )
        self.erode_slider.pack(side="left", padx=10)

        self.dilate_var = tk.IntVar(value=0)
        self.dilate_slider = tk.Scale(
            controls,
            from_=0,
            to=MAX_MORPH_ITERATIONS,
            orient="horizontal",
            variable=self.dilate_var,
            command=lambda _val: self.update_preview(),
            length=150,
            label="Dilate",
        )
        self.dilate_slider.pack(side="left", padx=10)

        self.save_button = tk.Button(controls, text="Save As...", command=self.save_image, state="disabled")
        self.save_button.pack(side="left")

        subtract_row = tk.Frame(root)
        subtract_row.pack(fill="x", padx=10, pady=(0, 8))

        self.subtract_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            subtract_row,
            text="Subtract dilated image from",
            variable=self.subtract_var,
            command=self.update_preview,
        ).pack(side="left")

        self.subtract_base_var = tk.StringVar(value="Grayscale original")
        tk.OptionMenu(
            subtract_row,
            self.subtract_base_var,
            "Grayscale original",
            "Binarized (pre-erode/dilate)",
            command=lambda _val: self.update_preview(),
        ).pack(side="left", padx=5)

        self.canvas = tk.Label(root)
        self.canvas.pack(padx=10, pady=10)

        if image_path:
            self.load_image(image_path)

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
                self.grayscale_img = img.convert("L")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return

        self.src_path = Path(path)
        self.root.title(f"Grayscale Threshold - {self.src_path.name}")
        self.save_button.config(state="normal")
        self.update_preview()

    def update_preview(self):
        if self.grayscale_img is None:
            return

        threshold = self.threshold_var.get()
        binarized = self.grayscale_img.point(lambda p: 255 if p >= threshold else 0, mode="L")
        pre_morph = binarized

        for _ in range(self.erode_var.get()):
            binarized = binarized.filter(ImageFilter.MinFilter(3))

        for _ in range(self.dilate_var.get()):
            binarized = binarized.filter(ImageFilter.MaxFilter(3))

        if self.subtract_var.get():
            base = self.grayscale_img if self.subtract_base_var.get() == "Grayscale original" else pre_morph
            result = ImageChops.subtract(base, binarized)
        else:
            result = binarized

        self.preview_img = result

        display_img = result.copy()
        display_img.thumbnail((MAX_PREVIEW_SIZE, MAX_PREVIEW_SIZE))
        self.tk_image = ImageTk.PhotoImage(display_img)
        self.canvas.config(image=self.tk_image)

    def save_image(self):
        if self.preview_img is None or self.src_path is None:
            return

        suffix_parts = [f"t{self.threshold_var.get()}", f"e{self.erode_var.get()}", f"d{self.dilate_var.get()}"]
        if self.subtract_var.get():
            base_tag = "gray" if self.subtract_base_var.get() == "Grayscale original" else "bin"
            suffix_parts.append(f"sub-{base_tag}")
        default_name = f"{self.src_path.stem}_{'_'.join(suffix_parts)}{self.src_path.suffix}"
        path = filedialog.asksaveasfilename(
            title="Save binarized image",
            initialfile=default_name,
            defaultextension=self.src_path.suffix,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.preview_img.save(path)
            messagebox.showinfo("Saved", f"Saved to {path}")


if __name__ == "__main__":
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None

    root = tk.Tk()
    app = ThresholdApp(root, initial_path)
    root.mainloop()
