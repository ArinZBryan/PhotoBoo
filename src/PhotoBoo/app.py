import logging
import os
from dataclasses import dataclass

from slimgui import imgui
from slimgui import implot

from PhotoBoo import imgui_window as imgui_windows

logger = logging.getLogger(__name__)

def run():
    window = imgui_windows.ImguiWindow(
        title="PhotoBoo",
        close_on_esc=False,
        font_bytes=None,
        request_opengl_core_profile=True,
    )
    window.set_font_size(24)

    while not window.should_close():
        window.begin_frame()

        for i in range(3):
            imgui.text("hello world!")

        res = imgui.button("click me")
        if res:
            logger.info("Clicked")

        window.end_frame()
    window.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    run()