"""Dataset definitions for the MLUA training scripts.

This module centralises the training and validation datasets that were
previously re-implemented in multiple entry points.  The implementation is
kept deliberately lightweight so it works on both Linux and Windows without
depending on any platform specific paths.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from evaluate.utils import get_data_test_overlap, rgb2gray


PathLike = Union[str, Path]


def _ensure_list(paths: Sequence[PathLike]) -> List[Path]:
    return [Path(p) for p in paths]


class TrainDataset(Dataset):
    """Shared training dataset used by all training scripts."""

    def __init__(
        self,
        image_list: Sequence[PathLike],
        label_list: Sequence[PathLike],
        ul_image_list: Optional[Sequence[PathLike]] = None,
        transize: int = 384,
    ) -> None:
        if len(image_list) != len(label_list):
            raise ValueError(
                "image_list and label_list must be the same length; got"
                f" {len(image_list)} and {len(label_list)}"
            )

        self.transize = transize
        self.samples: List[Tuple[Path, Optional[Path]]] = [
            (Path(img), Path(mask)) for img, mask in zip(image_list, label_list)
        ]

        if ul_image_list:
            self.samples.extend((Path(img), None) for img in ul_image_list)

        self.color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5)
        self.resize_transform = T.Resize((self.transize, self.transize))
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _shared_transforms(image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        angle = random.uniform(-45.0, 45.0)
        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]

        image = Image.open(image_path).convert("L")
        if label_path is not None:
            label = Image.open(label_path).convert("L")
        else:
            label = Image.new("L", image.size)

        image, label = self._shared_transforms(image, label)
        image = self.color_jitter(image)

        image = self.resize_transform(image)
        label = self.resize_transform(label)

        image_tensor = self.to_tensor(image).to(dtype=torch.float32)
        label_tensor = self.to_tensor(label).to(dtype=torch.float32)

        return image_tensor, label_tensor


class ValDataset(Dataset):
    """Validation dataset for panoramic evaluation."""

    def __init__(
        self,
        img_path_list: Sequence[PathLike],
        gt_path_list: Sequence[PathLike],
        patch_size: int = 384,
        stride: int = 192,
    ) -> None:
        if len(img_path_list) != len(gt_path_list):
            raise ValueError(
                "Validation image and ground-truth lists must have the same length;"
                f" got {len(img_path_list)} and {len(gt_path_list)}"
            )

        self.img_path_list = _ensure_list(img_path_list)
        self.gt_path_list = _ensure_list(gt_path_list)
        self.patch_size = patch_size
        self.stride = stride
        self.resize_transform = T.Resize((patch_size, patch_size))
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.img_path_list)

    @staticmethod
    def _normalize(inputs: np.ndarray) -> np.ndarray:
        return (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.img_path_list[index])
        gt_path = str(self.gt_path_list[index])
        patches, _, _, gt = get_data_test_overlap(
            img_path,
            gt_path,
            self.patch_size,
            self.patch_size,
            self.stride,
            self.stride,
        )
        patches = rgb2gray(patches)

        num_patches = patches.shape[0]
        final_img = torch.zeros(num_patches, self.patch_size, self.patch_size, dtype=torch.float32)
        for i in range(num_patches):
            patch = Image.fromarray(np.uint8(patches[i].squeeze()))
            patch = self.resize_transform(patch)
            final_img[i] = self.to_tensor(patch).squeeze(0)

        gt = self._normalize(gt)
        final_gt = torch.tensor(gt.squeeze(), dtype=torch.float32)

        return final_img, final_gt


__all__ = ["TrainDataset", "ValDataset"]
