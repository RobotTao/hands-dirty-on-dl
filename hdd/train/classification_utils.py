"""Classification train uitls."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def train_classifier_naive(
    net: nn.Module,
    criteria,
    optimizer: optim.optimizer.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Naive training procedure to train classifier for one epoch.

    Args:
        net (nn.Module): network instance.
        criteria (_type_): Loss function. Typically nn.CrossEntropyLoss
        optimizer (optim.optimizer.Optimizer): optimizer.
        train_loader (torch.utils.data.DataLoader): train data
        device (torch.device): device to run the training.

    Returns:
        avg train loss and train accuracy.
    """

    train_loss = 0.0
    correct_items = 0
    total_items = 0
    net.train()
    for Xs, ys in train_loader:
        Xs, ys = Xs.to(device), ys.to(device)
        optimizer.zero_grad()
        logits = net(Xs)
        loss = criteria(logits, ys)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct_items += torch.sum(torch.argmax(logits, dim=1) == ys).item()
        total_items += Xs.shape[0]

    avg_train_loss = train_loss / len(train_loader)
    accuracy = correct_items / total_items
    return avg_train_loss, accuracy
