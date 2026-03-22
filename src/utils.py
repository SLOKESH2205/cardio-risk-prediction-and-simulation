"""Shared utility helpers for file IO and serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def ensure_directory(path: Path) -> Path:
    """Create directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Created directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    """Convert values into JSON-safe types.

    Args:
        value: Arbitrary value.

    Returns:
        JSON-safe object.
    """
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    """Save JSON payload to disk.

    Args:
        path: Output file path.
        payload: Serializable object.

    Returns:
        None.
    """
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2)


def load_json(path: Path) -> Any:
    """Load JSON from disk.

    Args:
        path: Input file path.

    Returns:
        Parsed JSON payload.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_joblib(path: Path, obj: Any) -> None:
    """Save object using joblib.

    Args:
        path: Output file path.
        obj: Object to serialize.

    Returns:
        None.
    """
    ensure_directory(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Load object from joblib file.

    Args:
        path: Input file path.

    Returns:
        Deserialized object.
    """
    return joblib.load(path)


if __name__ == "__main__":
    demo_dir = ensure_directory(Path.cwd() / "outputs")
    save_json(demo_dir / "utils_demo.json", {"status": "ok"})
