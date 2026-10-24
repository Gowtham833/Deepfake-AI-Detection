# src/utils/io.py
"""I/O utilities for configuration, paths, and file operations."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Project root is resolved relative to this file:
# src/utils/io.py -> ../../ = project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return _PROJECT_ROOT


def load_config(config_path: str | Path | None = None) -> dict:
    """Load the YAML configuration file.

    Args:
        config_path: Path to config.yaml. If None, uses the default
                     location at configs/config.yaml relative to project root,
                     or the CONFIG_PATH environment variable.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if config_path is None:
        env_path = os.environ.get("CONFIG_PATH")
        if env_path:
            config_path = Path(env_path)
            if not config_path.is_absolute():
                config_path = _PROJECT_ROOT / config_path
        else:
            config_path = _PROJECT_ROOT / "configs" / "config.yaml"

    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Ensure configs/config.yaml exists or set CONFIG_PATH env var."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return config


def save_json(data: Any, path: str | Path) -> None:
    """Save data as a JSON file.

    Args:
        data: Data to serialize.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug("Saved JSON to %s", path)


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path: The resolved directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_environment_info() -> dict:
    """Collect environment information for reproducibility.

    Returns:
        dict: Python and key library versions.
    """
    import torch
    import torchvision
    import cv2
    import numpy as np

    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
