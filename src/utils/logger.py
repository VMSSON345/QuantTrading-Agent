import sys
from loguru import logger
from .paths import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(LOG_DIR / "app.log", level="DEBUG", rotation="10 MB", retention="7 days")

def get_logger(name: str = "quantagent"):
    return logger.bind(name=name)
