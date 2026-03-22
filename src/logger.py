"""Logging utilities for the cardiovascular risk platform."""

from __future__ import annotations

import logging
from pathlib import Path


class LoggerFactory:
    """Factory for consistent application loggers."""

    def __init__(self, log_dir: Path | None = None) -> None:
        """Initialize logger factory.

        Args:
            log_dir: Optional directory for log files.

        Returns:
            None.
        """
        self.log_dir = log_dir or Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "cardio_risk_platform.log"

    def get_logger(self, name: str) -> logging.Logger:
        """Return configured logger instance.

        Args:
            name: Logger name.

        Returns:
            Configured logger.
        """
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False
        return logger


_LOGGER_FACTORY = LoggerFactory()


def get_logger(name: str) -> logging.Logger:
    """Return shared project logger.

    Args:
        name: Logger name.

    Returns:
        Configured logger.
    """
    return _LOGGER_FACTORY.get_logger(name)


if __name__ == "__main__":
    demo_logger = get_logger(__name__)
    demo_logger.info("Logger demo executed successfully.")
