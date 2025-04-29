"""Classification train utils."""

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from hdd.train.early_stopping import EarlyStoppingInterface


def _train_classifier_naive(
    net: nn.Module,
    criteria,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Naive training procedure to train classifier for one epoch.

    Args:
        net: network instance.
        criteria: Loss function. Typically nn.CrossEntropyLoss
        optimizer: optimizer.
        train_loader: train data
        device: device to run the training.

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


def _eval_classifier_naive(
    net: nn.Module,
    criteria,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Naive eval procedure for classifier.

    Args:
        net: network instance.
        criteria: Loss function. Typically nn.CrossEntropyLoss
        val_loader: validation dataloader.
        device: device to run the training.

    Returns:
        avg val loss and val accuracy.
    """
    net.eval()
    correct_items = 0
    total_items = 0
    test_loss = 0.0
    with torch.no_grad():
        for Xs, ys in val_loader:
            Xs, ys = Xs.to(device), ys.to(device)
            logits = net(Xs)
            loss = criteria(logits, ys)
            correct_items += torch.sum(torch.argmax(logits, dim=1) == ys).item()
            total_items += Xs.shape[0]
            test_loss += loss.item()
    avg_val_loss = test_loss / len(val_loader)
    val_accuracy = correct_items / total_items
    return avg_val_loss, val_accuracy


def naive_train_classification_model(
    net: nn.Module,
    criteria,
    max_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    early_stopper: Optional[EarlyStoppingInterface] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Naive classifier training procedure.

    Args:
        net: classification model.
        criteria: loss function.
        max_epochs: maximum number of epochs.
        train_loader: train dataloader.
        val_loader: validation dataloader.
        device: network device.
        optimizer: optimizer
        scheduler: learning rate scheduler.
        early_stopper: early stopper.
        verbose: Print anything or not. Defaults to True.

    Returns:
        training statistics.
    """
    result = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        avg_train_loss, train_accuracy = _train_classifier_naive(
            net,
            criteria,
            optimizer,
            train_loader,
            device,
        )
        t1 = time.time()
        if scheduler is not None:
            scheduler.step()
        avg_val_loss, val_accuracy = _eval_classifier_naive(
            net,
            criteria,
            val_loader,
            device,
        )
        if verbose:
            print(
                f"Train Epoch: {epoch}/{max_epochs} "
                f"Avg Loss: {avg_train_loss:0.4f} "
                f"Accuracy: {train_accuracy:0.4f} "
                f"Time: {t1 - t0:0.5f} "
                f" | Val Avg Loss: {avg_val_loss:0.4f} "
                f"Accuracy: {val_accuracy:0.4f}"
            )
        result["train_loss"].append(avg_train_loss)
        result["val_loss"].append(avg_val_loss)
        result["train_accuracy"].append(train_accuracy)
        result["val_accuracy"].append(val_accuracy)
        if early_stopper is not None:
            if early_stopper(val_loss=avg_val_loss, model=net):
                early_stopper.load_best_model(net)
                return result

    return result
