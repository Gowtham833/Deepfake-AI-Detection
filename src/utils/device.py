# src/utils/device.py
"""Device detection and selection utilities."""

import os
import logging

import torch

logger = logging.getLogger(__name__)


def get_device(force: str | None = None) -> torch.device:
    """Get the best available compute device.

    Args:
        force: Force a specific device. One of "cuda", "cpu", or "auto"/None.

    Returns:
        torch.device: The selected device.
    """
    # Check environment variable override
    env_device = os.environ.get("DEVICE", "auto").lower()
    requested = (force or env_device).lower()

    if requested == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif requested == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU device (forced).")
    else:
        # Auto-detect
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Auto-detected CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            logger.info("Auto-detected CPU device.")

    return device


def get_device_info() -> dict:
    """Get detailed device information for reproducibility records.

    Returns:
        dict: Device information including type, name, and memory.
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": "cpu",
        "device_name": "CPU",
    }

    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 2), 1
        )

    return info
