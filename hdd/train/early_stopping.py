"""Early stopping utils.

This code is copied and modified from https://github.com/Bjarten/early-stopping-pytorch.
"""

from abc import ABC, abstractmethod
from io import BytesIO

import numpy as np
import torch


class EarlyStoppingInterface(ABC):
    @abstractmethod
    def __call__(self, val_loss, model) -> bool:
        """每个epoch之后调用该函数，如果可以早停，返回True."""
        pass

    @abstractmethod
    def load_best_model(self, net: torch.nn.Module) -> None:
        """如果早停止，可以从其获得最优模型至net中。如果没有早停就调用该函数，则触发异常。"""
        pass


class EarlyStoppingInMem(EarlyStoppingInterface):
    """Early stops the training if validation loss doesn't improve after a given patience.

    The best model is saved in Mem.
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.val_loss_min = np.inf
        self.delta = delta
        self.early_stop = False
        self.trace_func = trace_func
        self.best_model_in_mem = BytesIO()

    def __call__(self, val_loss, model) -> bool:
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return False

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self._save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self._save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def _save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # These two function calls reset the mem buffer.
        self.best_model_in_mem.seek(0)
        self.best_model_in_mem.truncate()
        torch.save(model.state_dict(), self.best_model_in_mem)
        self.val_loss_min = val_loss

    def load_best_model(self, net) -> None:
        """Load best model to net.

        Args:
            net: net to set best parameters.
            device: device to load model.

        Raises:
            RuntimeError: Called when not early stopped.
        """
        if not self.early_stop:
            raise RuntimeError(
                "You should not get the best model from non-early stop case."
            )
        self.best_model_in_mem.seek(0)

        net.load_state_dict(torch.load(self.best_model_in_mem, weights_only=True))


class EarlyStopping(EarlyStoppingInterface):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model) -> bool:
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return False

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self._save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self._save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def _save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_best_model(self, net: torch.nn.Module) -> None:
        if not self.early_stop:
            raise RuntimeError(
                "You should not get the best model from non-early stop case."
            )
        net.load_state_dict(torch.load(self.path, weights_only=True))
