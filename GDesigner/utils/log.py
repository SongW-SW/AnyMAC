import logging
import sys
import os
from pathlib import Path

# Create a logger
logger = logging.getLogger("GDesigner")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

# Optionally add file handler if log directory exists
log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "../logs"
if not log_dir.exists():
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "gdesigner.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create log directory: {e}")

# Export the logger
__all__ = ["logger"]
