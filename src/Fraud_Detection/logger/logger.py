import os
import logging
from datetime import datetime

LOG_DIR= "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)
log_level = logging.INFO
logger = logging.getLogger("Fraud_detection")
LOG_FILE_NAME = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
file_handler = logging.FileHandler(LOG_FILE_PATH)

stream_handler = logging.StreamHandler()
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
logger.setLevel(log_level)
file_handler.setFormatter(formatter)
stream_handler.setLevel(log_level)
stream_handler.setFormatter(formatter)
file_handler.setLevel(log_level)


