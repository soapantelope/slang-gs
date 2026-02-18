"""
The entry point for volumetric renderer.
"""

import asyncio
import logging
import os
import sys

from cs248a_renderer.gui.app import InteractiveRendererApp


log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)


if __name__ == "__main__":
    app = InteractiveRendererApp()
    asyncio.run(app.run())
