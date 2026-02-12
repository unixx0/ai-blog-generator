import logging

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Change to DEBUG if you want. Only show log messages that are this level or higher severity.

    if not logger.handlers:  # Avoid adding multiple handlers if imported multiple times
        # Console handler
        handler = logging.StreamHandler()  
        # Optional: save to file
        # handler = logging.FileHandler("app.log")

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
