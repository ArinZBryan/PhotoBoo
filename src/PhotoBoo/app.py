import argparse
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime

import cv2
import OpenGL.GL as gl
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

    def update(self):
        """Grab the latest frame into the texture. Returns True if a frame is available."""
        ok, frame = self.capture.read()
        if ok:
            # Swap on the BGR frame: the swapper's targets and MediaPipe input are BGR-based.
            frame = self.swapper.process(frame)
            self.last_frame = frame

            snap = self.swapper.pop_snapshot()
            if snap is not None and self.auto_save:
                save_capture(*snap)

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
        self.capture.release()
        self.swapper.close()
        gl.glDeleteTextures([self.texture])

class Border:
    """Draws a tiled picture-frame border from the tl/tm/tr/ml/mr/bl/bm/br images in a directory."""

    NAMES = ("tl", "tm", "tr", "ml", "mr", "bl", "bm", "br")

    def __init__(self, directory="border"):
        self.textures = {}
        self.sizes = {}
        for name in self.NAMES:
            path = Path(directory) / f"{name}.png"
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.error("Could not load border image %s", path)
                continue
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            height, width = image.shape[:2]
            texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            # Edge pieces are tiled by sampling past 1.0 in UV space
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image
            )
            self.textures[name] = texture
            self.sizes[name] = (width, height)

    def draw(self, draw_list, pos, size, scale):
        """Draw the border just inside the rectangle at pos with the given size."""
        if len(self.textures) < len(self.NAMES):
            return
        x0, y0 = pos
        x1, y1 = x0 + size[0], y0 + size[1]

        def dims(name):
            return self.sizes[name][0] * scale, self.sizes[name][1] * scale

        tl_w, tl_h = dims("tl")
        tr_w, tr_h = dims("tr")
        bl_w, bl_h = dims("bl")
        br_w, br_h = dims("br")

        # Edges first so the corners are drawn over any overlap
        top_len = (x1 - tr_w) - (x0 + tl_w)
        if top_len > 0:
            w, h = dims("tm")
            draw_list.add_image(self.textures["tm"], (x0 + tl_w, y0), (x1 - tr_w, y0 + h), (0, 0), (top_len / w, 1))
        bottom_len = (x1 - br_w) - (x0 + bl_w)
        if bottom_len > 0:
            w, h = dims("bm")
            draw_list.add_image(self.textures["bm"], (x0 + bl_w, y1 - h), (x1 - br_w, y1), (0, 0), (bottom_len / w, 1))
        left_len = (y1 - bl_h) - (y0 + tl_h)
        if left_len > 0:
            w, h = dims("ml")
            draw_list.add_image(self.textures["ml"], (x0, y0 + tl_h), (x0 + w, y1 - bl_h), (0, 0), (1, left_len / h))
        right_len = (y1 - br_h) - (y0 + tr_h)
        if right_len > 0:
            w, h = dims("mr")
            draw_list.add_image(self.textures["mr"], (x1 - w, y0 + tr_h), (x1, y1 - br_h), (0, 0), (1, right_len / h))

        draw_list.add_image(self.textures["tl"], (x0, y0), (x0 + tl_w, y0 + tl_h))
        draw_list.add_image(self.textures["tr"], (x1 - tr_w, y0), (x1, y0 + tr_h))
        draw_list.add_image(self.textures["bl"], (x0, y1 - bl_h), (x0 + bl_w, y1))
        draw_list.add_image(self.textures["br"], (x1 - br_w, y1 - br_h), (x1, y1))

    def close(self):
        gl.glDeleteTextures(list(self.textures.values()))

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
    webcam = WebcamFeed(args.camera)
    border = Border()

    show_settings = False

    while not window.should_close():
        window.begin_frame()

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