#!/usr/bin/env python3
"""Build a calibration set for Hailo quantization from the repo's stop-sign JPGs.

Produces calib.npy of shape (N, 640, 640, 3), uint8, RGB — the format
`hailo optimize --calib-set-path` expects. Run this on the x86 compile box
(or Colab) where the IMG_*.JPG files are available.
"""
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SIZE = 640


def letterbox(img, size=SIZE):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # standard YOLO pad
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def main():
    paths = sorted(glob.glob(os.path.join(REPO, "IMG_*.JPG")))
    if not paths:
        raise SystemExit("No IMG_*.JPG files found next to the repo root.")
    imgs = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print("skip unreadable:", p)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(letterbox(rgb))
    arr = np.stack(imgs).astype(np.uint8)
    out = os.path.join(HERE, "calib.npy")
    np.save(out, arr)
    print(f"Wrote {out}  shape={arr.shape} dtype={arr.dtype}  ({len(imgs)} images)")


if __name__ == "__main__":
    main()
