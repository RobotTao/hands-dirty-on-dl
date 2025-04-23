"""Util functions related to device."""

from typing import List, Union

import torch


def get_device(preferred_devices: Union[List[str], str]) -> torch.device:
    """Get first available device based on user preference.

    Args:
        preferred_devices: 按照偏好度有高到底排列的device序列。它可以是字符串或者
            字符串数组。

    Returns:
        First available device in the preferred_devices.
    """
    if isinstance(preferred_devices, str):
        preferred_devices = [preferred_devices]
    for preferred_device in preferred_devices:
        if preferred_device == "cuda" and torch.cuda.is_available():
            return torch.device(preferred_device)
        elif preferred_device == "mps" and torch.mps.is_available():
            return torch.device(preferred_device)
        elif preferred_device == "cpu":
            return torch.device(preferred_device)
        else:
            raise ValueError(
                f"Device {preferred_device} currently is not supported."
                "Please modify the code to enable it."
            )
    raise ValueError(f"No device is found.")
