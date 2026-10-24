import logging
import os
from dataclasses import dataclass

import cv2
import OpenGL.GL as gl
from slimgui import imgui
from slimgui import implot
from pathlib import Path

from PhotoBoo import imgui_window as imgui_windows
from PhotoBoo import faceswap as fs

logger = logging.getLogger(__name__)

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
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def update(self):
        """Grab the latest frame into the texture. Returns True if a frame is available."""
        ok, frame = self.capture.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#
            frame = self.swapper.process(frame)
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

@dataclass
class State:
    force_faceswap = False
    faceswap_gap = (1.0, 4.0)
    faceswap_time = 1.0
    faceswap_fade = 1.0
    

def run():
    state = State()

    window = imgui_windows.ImguiWindow(
        title="PhotoBoo",
        close_on_esc=False,
        font_bytes=None,
        request_opengl_core_profile=True,
    )
    window.set_font_size(24)
    webcam = WebcamFeed()

    show_settings = False

    while not window.should_close():
        window.begin_frame()

        if imgui.is_key_pressed(imgui.Key.KEY_F1, repeat=False):
            show_settings = not show_settings

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
        imgui.end()

        if show_settings:
            expanded, show_settings = imgui.begin("Settings", closable=True)
            if expanded:
                faceswap_now = imgui.button("Force Swap")
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


        window.end_frame()
    webcam.close()
    window.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    run()