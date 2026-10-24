# src/utils/__init__.py
"""Utility functions for the deepfake detection framework."""

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.io import load_config, get_project_root

__all__ = ["set_seed", "get_device", "load_config", "get_project_root"]
