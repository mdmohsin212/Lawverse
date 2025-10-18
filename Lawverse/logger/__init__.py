import logging
import os
from datetime import datetime

dir = '/tmp/logs'
# dir = '/logs'
os.makedirs(dir, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH = os.path.join(dir, LOG_FILE)

file_handler = logging.FileHandler(LOG_PATH)
console_handler = logging.StreamHandler()

log_format = "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler],
)

noisy_libs = [
    "urllib3",
    "requests",
    "httpx",
    "deep_translator",
    "PIL",
    "matplotlib",
]

for lib in noisy_libs:
    logging.getLogger(lib).setLevel(logging.WARNING)

logging.getLogger(__name__).info("✅ Logger initialized successfully with clean output.")