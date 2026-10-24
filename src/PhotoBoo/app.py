import argparse
import bisect
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import OpenGL.GL as gl
from PIL import Image, ImageSequence
from slimgui import imgui
from slimgui import implot
from pathlib import Path

from PhotoBoo import imgui_window as imgui_windows
from PhotoBoo import faceswap as fs

logger = logging.getLogger(__name__)

CAPTURE_DIR = Path("captures")


def save_capture(frame_bgr, label: str = "manual") -> None:
    """Write a BGR frame to captures/ as a timestamped JPEG, off the render thread."""
    CAPTURE_DIR.mkdir(exist_ok=True)
    path = CAPTURE_DIR / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{label}.jpg"

    def _write():
        if cv2.imwrite(str(path), frame_bgr):
            logger.info("Saved %s", path)
        else:
            logger.error("Failed to save %s", path)

    threading.Thread(target=_write, daemon=True).start()


class WebcamFeed:
    """Captures webcam frames with OpenCV and uploads them to an OpenGL texture."""

    def __init__(self, device=0):
        target_dir = Path("targets")
        paths = sorted(p for p in target_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.swapper = fs.FaceSwapper(paths, swap_len=2, gap=[1, 5], fade=0.2)
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            logger.error("Could not open webcam %s", device)
        self.texture = gl.glGenTextures(1)
        self.size = (0, 0)
        self.last_frame = None   # last displayed frame (BGR), for manual screenshots
        self.auto_save = False    # save one shot per swap
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        self._lock = threading.Lock()
        self._latest = None  # newest processed BGR frame not yet uploaded
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Read and face-swap frames off the render thread so slow swaps don't stall the UI."""
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Swap on the BGR frame: the swapper's targets and MediaPipe input are BGR-based.
            _t = time.monotonic()  # DIAGNOSTIC
            frame = self.swapper.process(frame)
            if time.monotonic() - _t > 0.1:
                logger.warning("DIAG slow swapper.process: %.0f ms", (time.monotonic() - _t) * 1e3)

            snap = self.swapper.pop_snapshot()
            if snap is not None and self.auto_save:
                save_capture(*snap)

            with self._lock:
                self._latest = frame

    def update(self):
        """Upload the newest processed frame into the texture. Returns True if a frame is available."""
        with self._lock:
            frame, self._latest = self._latest, None
        if frame is not None:
            self.last_frame = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame
            )
            self.size = (width, height)
        return self.size[0] > 0

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.capture.release()
        self.swapper.close()
        gl.glDeleteTextures([self.texture])

class BorderPiece:
    """One border image: a single texture for a still image, or one per frame for an animated GIF."""

    MIN_FRAME_SECONDS = 0.02  # GIFs with a 0 delay are conventionally played at ~10-20ms

    def __init__(self, path):
        self.textures = []
        self.frame_ends = []  # cumulative end time of each frame, in seconds
        with Image.open(path) as image:
            self.size = image.size
            for frame in ImageSequence.Iterator(image):
                duration = frame.info.get("duration", 0) / 1000
                rgba = np.ascontiguousarray(frame.convert("RGBA"))
                self.textures.append(self._upload(rgba))
                previous_end = self.frame_ends[-1] if self.frame_ends else 0.0
                self.frame_ends.append(previous_end + max(duration, self.MIN_FRAME_SECONDS))

    @staticmethod
    def _upload(rgba):
        height, width = rgba.shape[:2]
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        # Edge pieces are tiled by sampling past 1.0 in UV space
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba)
        return texture

    def texture_at(self, seconds):
        """The texture to show `seconds` into playback. Animations loop forever."""
        if len(self.textures) == 1:
            return self.textures[0]
        position = seconds % self.frame_ends[-1]
        return self.textures[bisect.bisect_right(self.frame_ends, position)]

    def close(self):
        gl.glDeleteTextures(self.textures)

class Border:
    """Draws a tiled picture-frame border from the tl/tm/tr/ml/mr/bl/bm/br images (PNG or GIF) in a directory."""

    NAMES = ("tl", "tm", "tr", "ml", "mr", "bl", "bm", "br")
    EXTENSIONS = (".gif", ".png")

    def __init__(self, directory="border"):
        self.pieces = {}
        self.start = time.monotonic()
        self._watch, self._watch_warned = {}, {}  # DIAGNOSTIC
        for name in self.NAMES:
            path = next((Path(directory) / f"{name}{ext}" for ext in self.EXTENSIONS
                         if (Path(directory) / f"{name}{ext}").exists()), None)
            if path is None:
                logger.error("No border image for %s in %s", name, directory)
                continue
            self.pieces[name] = BorderPiece(path)

    def draw(self, draw_list, pos, size, scale):
        """Draw the border just inside the rectangle at pos with the given size."""
        if len(self.pieces) < len(self.NAMES):
            return
        now = time.monotonic() - self.start
        texture = {name: piece.texture_at(now) for name, piece in self.pieces.items()}
        # DIAGNOSTIC: report an animated piece whose frame hasn't changed for a while
        for name, piece in self.pieces.items():
            last_tex, since = self._watch.get(name, (None, now))
            if texture[name] != last_tex:
                self._watch[name] = (texture[name], now)
            elif len(piece.textures) > 1 and now - since > 0.5 and now - self._watch_warned.get(name, -9) > 2:
                self._watch_warned[name] = now
                logger.warning("DIAG border %s stuck on texture %s for %.1fs", name, texture[name], now - since)
        x0, y0 = pos
        x1, y1 = x0 + size[0], y0 + size[1]

        def dims(name):
            return self.pieces[name].size[0] * scale, self.pieces[name].size[1] * scale

        tl_w, tl_h = dims("tl")
        tr_w, tr_h = dims("tr")
        bl_w, bl_h = dims("bl")
        br_w, br_h = dims("br")

        # Edges first so the corners are drawn over any overlap
        top_len = (x1 - tr_w) - (x0 + tl_w)
        if top_len > 0:
            w, h = dims("tm")
            draw_list.add_image(texture["tm"], (x0 + tl_w, y0), (x1 - tr_w, y0 + h), (0, 0), (top_len / w, 1))
        bottom_len = (x1 - br_w) - (x0 + bl_w)
        if bottom_len > 0:
            w, h = dims("bm")
            draw_list.add_image(texture["bm"], (x0 + bl_w, y1 - h), (x1 - br_w, y1), (0, 0), (bottom_len / w, 1))
        left_len = (y1 - bl_h) - (y0 + tl_h)
        if left_len > 0:
            w, h = dims("ml")
            draw_list.add_image(texture["ml"], (x0, y0 + tl_h), (x0 + w, y1 - bl_h), (0, 0), (1, left_len / h))
        right_len = (y1 - br_h) - (y0 + tr_h)
        if right_len > 0:
            w, h = dims("mr")
            draw_list.add_image(texture["mr"], (x1 - w, y0 + tr_h), (x1, y1 - br_h), (0, 0), (1, right_len / h))

        draw_list.add_image(texture["tl"], (x0, y0), (x0 + tl_w, y0 + tl_h))
        draw_list.add_image(texture["tr"], (x1 - tr_w, y0), (x1, y0 + tr_h))
        draw_list.add_image(texture["bl"], (x0, y1 - bl_h), (x0 + bl_w, y1))
        draw_list.add_image(texture["br"], (x1 - br_w, y1 - br_h), (x1, y1))

    def close(self):
        for piece in self.pieces.values():
            piece.close()

@dataclass
class State:
    force_faceswap = False
    faceswap_gap = (1.0, 4.0)
    faceswap_time = 1.0
    faceswap_fade = 0.3
    border_scale = 1.0
    blink_mode = False        # swap when someone blinks instead of on a timer
    blink_threshold = 0.5
    

def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="photoboo", description="The haunted-house mirror from hell")
    parser.add_argument(
        "-c", "--camera", type=int, default=0, metavar="INDEX",
        help="index of the camera device to use (default: 0)",
    )
    return parser.parse_args(argv)

def run(argv=None):
    args = parse_args(argv)
    state = State()

    window = imgui_windows.ImguiWindow(
        title="PhotoBoo",
        close_on_esc=False,
        font_bytes=None,
        request_opengl_core_profile=True,
    )
    window.set_font_size(24)
    window.set_vsync(True)  # the webcam no longer paces the loop, so cap it to the display
    webcam = WebcamFeed(args.camera)
    border = Border()

    show_settings = False
    last_frame_time = time.monotonic()  # DIAGNOSTIC


    while not window.should_close():
        window.begin_frame()
        _now = time.monotonic()  # DIAGNOSTIC
        if _now - last_frame_time > 0.06:
            logger.warning("DIAG slow render frame: %.0f ms (swapping=%s)", (_now - last_frame_time) * 1e3, webcam.swapper.swapping)
        last_frame_time = _now

        if imgui.is_key_pressed(imgui.Key.KEY_F1, repeat=False):
            show_settings = not show_settings
        if imgui.is_key_pressed(imgui.Key.KEY_F2, repeat=False) and webcam.last_frame is not None:
            save_capture(webcam.last_frame.copy())

        # The main "Debug" window covers the whole viewport
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.pos, imgui.Cond.ALWAYS)
        imgui.set_next_window_size(viewport.size, imgui.Cond.ALWAYS)
        imgui.push_style_var(imgui.StyleVar.WINDOW_PADDING, (0, 0))
        visible, _ = imgui.begin(
            "Debug",
            flags=imgui.WindowFlags.NO_DECORATION
            | imgui.WindowFlags.NO_MOVE
            | imgui.WindowFlags.NO_SAVED_SETTINGS
            | imgui.WindowFlags.NO_BRING_TO_FRONT_ON_FOCUS,
        )
        imgui.pop_style_var()
        if visible:
            if webcam.update():
                # Stretch the feed to fill the whole window, ignoring aspect ratio
                imgui.image(int(webcam.texture), imgui.get_content_region_avail())
            else:
                imgui.text("No webcam feed")
            border.draw(imgui.get_window_draw_list(), viewport.pos, viewport.size, state.border_scale)
        imgui.end()

        if show_settings:
            expanded, show_settings = imgui.begin("Settings", closable=True)
            if expanded:
                _, state.border_scale = imgui.slider_float("Border scale", state.border_scale, 0.1, 10.0)
                state.force_faceswap = imgui.button("Force Swap")
                _, webcam.auto_save = imgui.checkbox("Save photos", webcam.auto_save)
                _, state.blink_mode = imgui.checkbox("Swap on blink (replaces timer)", state.blink_mode)

                if state.blink_mode:
                    _, state.blink_threshold = imgui.slider_float(
                        "Blink threshold", state.blink_threshold, 0.1, 0.9, "%.2f"
                    )
                faceswap_gap, state.faceswap_gap = imgui.input_float2("Time between swaps (seconds)", state.faceswap_gap)
                faceswap_fade, state.faceswap_fade = imgui.input_float("Faceswap fade time (seconds)", state.faceswap_fade)
                faceswap_len, state.faceswap_time = imgui.input_float("Faceswap time (seconds)", state.faceswap_time)
            imgui.end()

        """options:
        time during swap (swap_len) float
        fade time of swap (fade) float
        gap between swaps (gap) (tuple[float, float])
        """


        if state.force_faceswap:
            webcam.swapper.trigger()
        if state.faceswap_fade < 0.00001:
            state.faceswap_fade = 0.00001
        webcam.swapper.fade = state.faceswap_fade
        webcam.swapper.swap_len = state.faceswap_time
        webcam.swapper.gap = state.faceswap_gap
        webcam.swapper.blink_mode = state.blink_mode
        webcam.swapper.blink_threshold = state.blink_threshold


        window.end_frame()
    border.close()
    webcam.close()
    window.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    run()