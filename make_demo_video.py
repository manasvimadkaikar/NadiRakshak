"""
Creates two demo CCTV-style videos from a canoe photo, so you can test
without real footage:

  demo_stationary_night.mp4 -> canoe stays in one place, dark (night) footage
  demo_moving_day.mp4       -> canoe keeps moving across the frame (fishing / transit)

Usage:
    python make_demo_video.py                 (uses sample_canoe.jpg)
    python make_demo_video.py my_photo.jpg
"""
import sys

import cv2
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "sample_canoe.jpg"
img = cv2.imread(src)
if img is None:
    sys.exit(f"Could not read {src}")
img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])))
H, W = img.shape[:2]
OUT_W, OUT_H, FPS = 960, 540, 25


def write(name, frames_fn, seconds):
    vw = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    for i in range(seconds * FPS):
        vw.write(frames_fn(i))
    vw.release()
    print("created", name)


rng = np.random.default_rng(0)


def stationary_night(i):
    # small camera shake + darkness + sensor noise = realistic night CCTV
    dx = int(6 * np.sin(i / 17)); dy = int(4 * np.cos(i / 23))
    x0 = (W - OUT_W) // 2 + dx; y0 = max(0, (H - OUT_H) // 2 + dy)
    f = img[y0:y0 + OUT_H, x0:x0 + OUT_W].copy()
    f = cv2.resize(f, (OUT_W, OUT_H))
    f = (f * 0.28).astype(np.uint8)                              # make it dark
    noise = rng.normal(0, 6, f.shape).astype(np.int16)
    f = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.putText(f, "CAM-01  REC", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return f


def moving_day(i, seconds=30):
    # camera pans across the scene, so the canoe keeps moving in the frame
    span = W - OUT_W
    x0 = int(span * (0.5 - 0.5 * np.cos(2 * np.pi * i / (8 * FPS))))   # pans back and forth
    y0 = max(0, (H - OUT_H) // 2)
    f = cv2.resize(img[y0:y0 + OUT_H, x0:x0 + OUT_W], (OUT_W, OUT_H))
    cv2.putText(f, "CAM-02  REC", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return f


write("demo_stationary_night.mp4", stationary_night, 30)
write("demo_moving_day.mp4", moving_day, 30)
