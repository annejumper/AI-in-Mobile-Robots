"""Generate a printable AprilTag image using OpenCV's aruco module.

Usage:
    python generate_tag.py [--id ID] [--dict DICT_NAME] [--size PIXELS] [--margin PIXELS] [--out PATH]

Example:
    python generate_tag.py --id 0 --size 600 --out tags/tag_0.png
"""

import argparse
import os

import cv2
import numpy as np

DEFAULT_DICT = "DICT_APRILTAG_36h11"


def generate_tag(tag_id: int, dict_name: str, tag_pixels: int, margin_pixels: int) -> np.ndarray:
    dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    tag_img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, tag_pixels)

    # Add a white quiet-zone border. AprilTag detectors rely on contrast
    # between the tag and its surroundings, so printed tags need margin.
    bordered = cv2.copyMakeBorder(
        tag_img,
        margin_pixels, margin_pixels, margin_pixels, margin_pixels,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    return bordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=0, help="Tag ID to encode (default: 0)")
    parser.add_argument("--dict", type=str, default=DEFAULT_DICT,
                         help=f"ArUco/AprilTag dictionary name (default: {DEFAULT_DICT})")
    parser.add_argument("--size", type=int, default=600, help="Tag size in pixels, excluding margin (default: 600)")
    parser.add_argument("--margin", type=int, default=80, help="White border margin in pixels (default: 80)")
    parser.add_argument("--out", type=str, default=None,
                         help="Output file path (default: tags/tag_<id>_<dict>.png)")
    args = parser.parse_args()

    if args.out is None:
        os.makedirs("tags", exist_ok=True)
        args.out = os.path.join("tags", f"tag_{args.id}_{args.dict}.png")
    else:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    img = generate_tag(args.id, args.dict, args.size, args.margin)
    cv2.imwrite(args.out, img)
    print(f"Saved tag id={args.id} dict={args.dict} to {args.out} ({img.shape[1]}x{img.shape[0]} px)")
    print("Print this at a known physical size (e.g. measure the black square's side in cm) "
          "-- you'll need that measurement later for distance estimation.")


if __name__ == "__main__":
    main()
