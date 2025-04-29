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
