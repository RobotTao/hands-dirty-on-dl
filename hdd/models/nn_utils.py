from typing import Optional

import torch.nn as nn


def activation_from_name(name: str) -> nn.Module:
    """Util function to get activation layer based on name.

    This function is just to reduce the burden when we create
    activation function based on the name.

    Args:
        name: activation function name.

    Returns:
        activation function nn module.
    """

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"activation {name} is not supported. Please update code.")


def count_trainable_parameter(model: nn.Module) -> int:
    """Get the total number trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
