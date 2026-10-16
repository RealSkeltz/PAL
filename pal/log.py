import logging

from pal.config import LOGS_DIR

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(LOGS_DIR / "pal.log"), logging.StreamHandler()],
)
