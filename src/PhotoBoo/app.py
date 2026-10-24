import logging
import os
from dataclasses import dataclass

import cv2
import OpenGL.GL as gl
from slimgui import imgui
from slimgui import implot

from PhotoBoo import imgui_window as imgui_windows

logger = logging.getLogger(__name__)

class WebcamFeed:
    """Captures webcam frames with OpenCV and uploads them to an OpenGL texture."""

    def __init__(self, device=0):
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
        gl.glDeleteTextures([self.texture])

def run():
    window = imgui_windows.ImguiWindow(
        title="PhotoBoo",
        close_on_esc=False,
        font_bytes=None,
        request_opengl_core_profile=True,
    )
    window.set_font_size(24)
    webcam = WebcamFeed()

    while not window.should_close():
        window.begin_frame()

        if webcam.update():
            # Stretch the feed to fill the whole window, ignoring aspect ratio
            imgui.image(int(webcam.texture), imgui.get_content_region_avail())
        else:
            imgui.text("No webcam feed")
            
        window.end_frame()
    webcam.close()
    window.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    run()