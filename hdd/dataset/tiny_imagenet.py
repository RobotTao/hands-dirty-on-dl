"""这个代码是从torchvision的Imagenette代码修改而来."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


def _load_train_samples(
    wnid_to_idx: Dict[str, int], train_root: Path
) -> List[Tuple[str, int]]:
    """Load train samples."""
    instances = []
    for wnid, class_index in wnid_to_idx.items():
        class_folder = train_root / Path(wnid) / "images"
        for root, _, fnames in sorted(os.walk(class_folder, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if has_file_allowed_extension(fname, "jpeg"):
                    instances.append((path, class_index))

    return instances


def _load_val_samples(
    wnid_to_idx: Dict[str, int], val_root: Path
) -> List[Tuple[str, int]]:
    """Load val samples."""
    instances = []
    with open(val_root / Path("val_annotations.txt"), "r") as f:
        for line in f.readlines():
            items = line.split("\t")
            filename, wnid = items[0], items[1]
            class_index = wnid_to_idx[wnid]
            path = os.path.join(val_root, "images", filename)
            instances.append((path, class_index))
    return instances


class TinyImagenet(VisionDataset):
    """Tiny Imagenet.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Tiny Imagenet dataset.
        split (string, optional): The dataset split. Supports ``"train"`` (default) and ``"val"``.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version, e.g. ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class name, class index).
        wnid_to_idx (dict): Dict with items (WordNet ID, class index).
    """

    _ARCHIVES = (
        "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "90528d7ca1a48142e341f4ef8d21d0de",
    )

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        download=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ["train", "val"])
        self._url, self._md5 = self._ARCHIVES
        self._image_root = Path(self.root) / Path(self._url).stem
        self._train_root = self._image_root / Path("train")
        self._val_root = self._image_root / Path("val")
        self._test_root = self._image_root / Path("test")
        if download:
            self._download()
        elif not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        self.wnids, self.classes, self.wnid_to_idx = self._load_wnids_and_classes()
        if self._split == "train":
            self._samples = _load_train_samples(self.wnid_to_idx, self._train_root)
        elif self._split == "val":
            self._samples = _load_val_samples(self.wnid_to_idx, self._val_root)
        self.targets = [target for _, target in self._samples]
        # Preload data into memory
        self._loaded_images = []
        for path, _ in self._samples:
            image = Image.open(path).convert("RGB")
            self._loaded_images.append(image)

    def _load_wnids_and_classes(
        self,
    ) -> tuple[list[Any], list[Any], dict[Any, Any]]:

        all_wnids_to_names = {}
        with open(self._image_root / Path("words.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip()
                wnid = line[: line.find("\t")]
                names = line[line.find("\t") + 1 :]
                names = names.split(",")
                all_wnids_to_names[wnid] = names

        wnids = []
        classes = []
        wnid_to_idx = {}
        with open(self._image_root / Path("wnids.txt"), "r") as f:
            idx = 0
            for line in f.readlines():
                wnid = line.strip()
                wnids.append(wnid)
                classes.append(all_wnids_to_names[wnid])
                wnid_to_idx[wnid] = idx
                idx += 1
            assert idx == 200
        return wnids, classes, wnid_to_idx

    def _check_exists(self) -> bool:
        return self._image_root.exists()

    def _download(self):
        if not self._check_exists():
            download_and_extract_archive(self._url, self.root, md5=self._md5)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        target = self.targets[idx]
        image = self._loaded_images[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self._samples)
