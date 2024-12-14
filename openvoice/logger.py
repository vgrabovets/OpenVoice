import logging
import os

from termcolor import colored


class CustomFormatter(logging.Formatter):
    _format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: colored(_format, "white"),
        logging.INFO: colored(_format, "cyan"),
        logging.WARNING: colored(_format, "yellow"),
        logging.ERROR: colored(_format, "red"),
        logging.CRITICAL: colored(_format, "red", attrs=["bold"]),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


if os.getenv("PYTHON_LOG_DEBUG"):
    level = logging.DEBUG
else:
    level = logging.INFO

logging.getLogger().setLevel(level)
log_handler = logging.StreamHandler()
# log_handler.setLevel(level)
log_handler.setFormatter(CustomFormatter())

logger = logging.getLogger("logger")
logger.propagate = False
logger.addHandler(log_handler)
logger.setLevel(level)

url_lib_logger = logging.getLogger("urllib3")
url_lib_logger.setLevel(logging.WARNING)

asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.setLevel(logging.WARNING)
