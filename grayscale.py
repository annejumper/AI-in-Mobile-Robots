import sys
from pathlib import Path

from PIL import Image


def convert_to_grayscale(input_path: str, output_path: str | None = None) -> str:
    src = Path(input_path)
    dest = Path(output_path) if output_path else src.with_stem(f"{src.stem}_grayscale")

    with Image.open(src) as img:
        grayscale_img = img.convert("L")
        grayscale_img.save(dest)

    return str(dest)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grayscale.py <input_image> [output_image]")
        sys.exit(1)

    input_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else None

    result_path = convert_to_grayscale(input_arg, output_arg)
    print(f"Saved grayscale image to {result_path}")
