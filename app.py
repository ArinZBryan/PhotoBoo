import logging
import os
from dataclasses import dataclass

from slimgui import imgui
from slimgui import implot



def run():
    window = imgui_windows.ImguiWindow(
        title="PhotoBoo",
        close_on_esc=False,
        font_bytes=None,
        request_opengl_core_profile=True,
    )
    window.set_font_size(24)
    

if __name__ == "__main__":
    logging.basicConfig(
        level.os.environ.get("LOGLEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    run()