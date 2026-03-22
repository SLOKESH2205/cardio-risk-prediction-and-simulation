"""Custom exception types for the cardiovascular risk platform."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CustomException(Exception):
    """Custom exception carrying file and context information."""

    message: str

    def __str__(self) -> str:
        """Return string representation.

        Args:
            None.

        Returns:
            Error message text.
        """
        return self.message


if __name__ == "__main__":
    raise SystemExit("CustomException module loaded successfully.")
