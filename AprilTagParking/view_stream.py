"""Quick sanity check that OpenCV can read your camera source.

Usage:
    # Phone running an IP-camera app over Wi-Fi:
    python view_stream.py --url http://192.168.1.42:8080/video

    # iPhone via macOS Continuity Camera (or any local webcam), by device index:
    python view_stream.py --camera-index 1

    # Not sure which index is the iPhone? List and preview all of them:
    python view_stream.py --list

Press 'q' to quit.
"""

import argparse
import sys

import cv2

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def list_cameras(max_index=5):
    print("Probing camera indices 0-%d. A window will open for each one found;" % (max_index - 1))
    print("press 'q' to move on to the next index.")
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok:
            cap.release()
            continue
        print(f"Index {index}: opened successfully, frame size {frame.shape[1]}x{frame.shape[0]}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"index {index} -- press 'q' for next", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera index probe", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="Phone IP-camera stream URL")
    source_group.add_argument("--camera-index", type=int, help="Local camera device index (e.g. Continuity Camera)")
    source_group.add_argument("--list", action="store_true", help="Probe and preview camera indices 0-4 to find the right one")
    parser.add_argument("--rotate", type=int, default=0, choices=sorted(ROTATIONS), help="Rotate the feed clockwise by this many degrees (default 0)")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    source = args.url if args.url is not None else args.camera_index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source {source}")

    rotation = ROTATIONS[args.rotate]

    print("Streaming. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame -- stream may have dropped.")
            break
        if rotation is not None:
            frame = cv2.rotate(frame, rotation)
        cv2.imshow("Camera Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
