import random
from typing import Any, Dict, List, TypeVar

import torch
import torchvision
from torchvision.transforms import functional as F


class RandomResize(torch.nn.Module):
    """Randomly select a number from sizes and use transforms.functional.resize to resize it."""

    def __init__(
        self,
        sizes: List[int],
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True,
    ):
        super().__init__()

        self.sizes = sizes
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        size = random.choice(self.sizes)
        return F.resize(img, size, self.interpolation, self.max_size, self.antialias)

    def __repr__(self) -> str:
        detail = f"(size={self.sizes}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class RandomMetaTransform:
    """Randomly select a transform from list of transforms."""

    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def __call__(self, pic) -> torch.Tensor:
        transform = random.choice(self.transforms)
        return transform(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


T = TypeVar("T")


class TargetLabelRemapping:
    """Remap the target label to another value.

    One typical usage is transform imagenette label to imagenet label using pre-trained models.
    """

    def __init__(self, target_map: Dict[T, T]):
        self.target_map = target_map

    def __call__(self, y: T) -> T:
        return self.target_map[y]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
