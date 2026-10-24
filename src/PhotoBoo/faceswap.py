"""Live face swap with periodic random swaps.

MediaPipe Tasks (FaceLandmarker) for landmarks, per-triangle affine warp,
cv2.seamlessClone (Poisson blending) for the composite.
"""
from __future__ import annotations

import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from scipy.spatial import Delaunay

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")


# --------------------------------------------------------------------------- landmarks

def ensure_model(path: Path = MODEL_PATH) -> Path:
    if not path.exists():
        print(f"Downloading face landmarker model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def make_landmarker(mode: vision.RunningMode) -> vision.FaceLandmarker:
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=mode,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def to_mp_image(img_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def result_to_points(result, shape) -> np.ndarray | None:
    if not result.face_landmarks:
        return None
    h, w = shape[:2]
    return np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], np.float32)


# --------------------------------------------------------------------------- targets

@dataclass
class Target:
    name: str
    image: np.ndarray       # BGR
    points: np.ndarray      # (N, 2) float32, pixel coords
    triangles: np.ndarray   # (T, 3) indices into points


def load_target(path: str | Path, landmarker: vision.FaceLandmarker) -> Target:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    pts = result_to_points(landmarker.detect(to_mp_image(img)), img.shape)
    if pts is None:
        raise ValueError(f"No face found in {path}")
    h, w = img.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return Target(Path(path).stem, img, pts, Delaunay(pts).simplices)


# --------------------------------------------------------------------------- swapping

def warp_triangle(src: np.ndarray, dst: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> None:
    """Affine-warp triangle t_src of src onto triangle t_dst of dst, in place."""
    x1, y1, w1, h1 = cv2.boundingRect(t_src)
    x2, y2, w2, h2 = cv2.boundingRect(t_dst)
    H, W = dst.shape[:2]
    if min(w1, h1, w2, h2) == 0 or x2 < 0 or y2 < 0 or x2 + w2 > W or y2 + h2 > H:
        return

    o1 = np.array([x1, y1], np.float32)
    o2 = np.array([x2, y2], np.float32)
    M = cv2.getAffineTransform(t_src - o1, t_dst - o2)
    patch = cv2.warpAffine(
        src[y1:y1 + h1, x1:x1 + w1], M, (w2, h2),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((h2, w2), np.uint8)
    cv2.fillConvexPoly(mask, (t_dst - o2).astype(np.int32), 255)
    roi = dst[y2:y2 + h2, x2:x2 + w2]
    roi[mask > 0] = patch[mask > 0]


def swap_face(frame: np.ndarray, target: Target, pts: np.ndarray,
              clone_mode: int = cv2.NORMAL_CLONE) -> np.ndarray:
    """Warp target's face onto the face at `pts` in `frame` and Poisson-blend it."""
    warped = frame.copy()
    for tri in target.triangles:
        warp_triangle(target.image, warped, target.points[tri], pts[tri])

    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))  # seamlessClone fails if mask hits the border
    if not mask.any():
        return frame

    x, y, w, h = cv2.boundingRect(mask)
    try:
        return cv2.seamlessClone(warped, frame, mask, (x + w // 2, y + h // 2), clone_mode)
    except cv2.error:
        return frame


class FaceSwapper:
    """Call process(frame) on every frame; swaps happen at random intervals.

    Use trigger() to force a swap now (e.g. when Gemma picks a target).
    """

    def __init__(self, target_paths, swap_len: float = 3.0, fade: float = 0.4,
                 gap: tuple[float, float] = (5.0, 15.0)):
        with make_landmarker(vision.RunningMode.IMAGE) as lm:
            self.targets = [load_target(p, lm) for p in target_paths]
        if not self.targets:
            raise ValueError("No target images given")

        self.live = make_landmarker(vision.RunningMode.VIDEO)
        self.swap_len, self.fade, self.gap = swap_len, fade, gap
        self._last_ts = -1
        self._schedule(time.monotonic())

    def _schedule(self, now: float) -> None:
        self.swap_start = now + random.uniform(*self.gap)
        self.current = random.choice(self.targets)

    def trigger(self, target_name: str | None = None) -> None:
        """Start a swap immediately, optionally with a specific target (by file stem)."""
        if target_name is not None:
            matches = [t for t in self.targets if t.name == target_name]
            if not matches:
                raise KeyError(target_name)
            self.current = matches[0]
        self.swap_start = time.monotonic()

    def _detect(self, frame: np.ndarray) -> np.ndarray | None:
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # VIDEO mode needs increasing timestamps
        self._last_ts = ts
        return result_to_points(self.live.detect_for_video(to_mp_image(frame), ts), frame.shape)

    @property
    def swapping(self) -> bool:
        return 0.0 <= time.monotonic() - self.swap_start <= self.swap_len

    def process(self, frame: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        t = now - self.swap_start
        if t > self.swap_len:
            self._schedule(now)
            return frame
        if t < 0:
            return frame

        pts = self._detect(frame)
        if pts is None:
            return frame

        alpha = min(1.0, t / self.fade, (self.swap_len - t) / self.fade)
        swapped = swap_face(frame, self.current, pts)
        return cv2.addWeighted(swapped, alpha, frame, 1.0 - alpha, 0)

    def close(self) -> None:
        self.live.close()


# --------------------------------------------------------------------------- demo

def main() -> None:
    target_dir = Path("targets")
    paths = sorted(p for p in target_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    swapper = FaceSwapper(paths)
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror
            cv2.imshow("PhotoBoo", swapper.process(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:          # Esc: quit
                break
            if key == ord(" "):    # Space: swap now
                swapper.trigger()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        swapper.close()


if __name__ == "__main__":
    main()