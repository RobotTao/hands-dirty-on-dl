"""这个代码是从torchvision的Imagenette代码修改而来，它将图片数据提前载入内存中，可以有效提高训练速度。"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.models import VGG19_BN_Weights


class ImagenetteInMemory(VisionDataset):
    """`Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ image classification dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Imagenette dataset.
        split (string, optional): The dataset split. Supports ``"train"`` (default), and ``"val"``.
        size (string, optional): The image size. Supports ``"full"`` (default), ``"320px"``, and ``"160px"``.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version, e.g. ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class name, class index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (WordNet ID, class index).
    """

    _ARCHIVES = {
        "full": (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
            "fe2fc210e6bb7c5664d602c3cd71e612",
        ),
        "320px": (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
            "3df6f0d01a2c9592104656642f5e78a3",
        ),
        "160px": (
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
            "e793b78cc4c9e9a4ccc0c1155377a412",
        ),
    }
    _WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chain saw", "chainsaw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        size: str = "full",
        download=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ["train", "val"])
        self._size = verify_str_arg(size, "size", ["full", "320px", "160px"])

        self._url, self._md5 = self._ARCHIVES[self._size]
        self._size_root = Path(self.root) / Path(self._url).stem
        self._image_root = str(self._size_root / self._split)

        if download:
            self._download()
        elif not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        self.wnids, self.wnid_to_idx = find_classes(self._image_root)
        self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            class_name: idx
            for wnid, idx in self.wnid_to_idx.items()
            for class_name in self._WNID_TO_CLASS[wnid]
        }
        self._samples = make_dataset(
            self._image_root, self.wnid_to_idx, extensions=".jpeg"
        )
        self._loaded_images = []
        for path, _ in self._samples:
            image = Image.open(path).convert("RGB")
            self._loaded_images.append(image)

    def _check_exists(self) -> bool:
        return self._size_root.exists()

    def _download(self):
        if not self._check_exists():
            download_and_extract_archive(self._url, self.root, md5=self._md5)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        _, label = self._samples[idx]
        image = self._loaded_images[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)


def get_mean_and_std(dataset: ImagenetteInMemory):
    """Compute the mean and std of dataset.

    Args:
        dataset (ImagenetteInMemory): Imagenette dataset.
    """
    # Compute train data mean and std
    # Note: we just compute the mean of each image's mean and std.
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for i in range(len(dataset)):
        I, _ = dataset[i]
        mean += torch.mean(I, dim=(1, 2))
        std += torch.std(I, dim=(1, 2))
    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std


def get_imagenette_label_to_imagenet_label() -> Dict[int, int]:
    imagenette_label_to_imagenet_label = {}
    imagenette_class_names = [
        "tench",
        "English springer",
        "cassette player",
        "chain saw",
        "church",
        "French horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ]
    for imagenette_label, imagenette_label_name in enumerate(imagenette_class_names):
        imagenet_label = VGG19_BN_Weights.IMAGENET1K_V1.meta["categories"].index(
            imagenette_label_name
        )
        imagenette_label_to_imagenet_label[imagenette_label] = imagenet_label
    return imagenette_label_to_imagenet_label
