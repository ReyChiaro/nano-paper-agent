# paper_agent/utils/logger.py

import logging
import os
from .config import config  # Import our config instance


class AppLogger:
    """
    Centralized logging utility for the Paper Agent application.
    Configures logging based on settings in config.py.
    """

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppLogger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        """
        Sets up the logging configuration.
        """
        if self._logger is not None:
            return  # Already set up

        self._logger = logging.getLogger("PaperAgent")
        log_level_str = config.get("LOG_LEVEL", "INFO").upper()
        log_file = config.get("LOG_FILE")

        # Set the logging level
        numeric_level = getattr(logging, log_level_str, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level_str}")
        self._logger.setLevel(numeric_level)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        # Prevent duplicate logs if handlers are added multiple times
        self._logger.propagate = False

    def get_logger(self):
        """
        Returns the configured logger instance.
        """
        return self._logger


# Initialize the logger instance for easy access
logger = AppLogger().get_logger()

if __name__ == "__main__":
    # Example usage and testing the logger module
    print("--- Testing Logger Module ---")
    # You can import 'logger' directly in other modules
    # from utils.logger import logger

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    print(f"\nCheck '{config.get('LOG_FILE')}' for file logs.")
