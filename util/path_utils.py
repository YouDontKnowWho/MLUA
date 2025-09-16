"""Helper utilities for locating dataset assets on different platforms."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

PathLike = Union[str, Path]


def _numeric_sort_key(path: Path) -> tuple[int, Union[int, str]]:
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def _list_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted([p for p in directory.iterdir() if p.is_file()], key=_numeric_sort_key)


def collect_training_paths(root: PathLike) -> Tuple[List[Path], List[Path], List[Path]]:
    root_path = Path(root)
    image_dir = root_path / "train" / "images"
    label_dir = root_path / "train" / "labels"
    unlabeled_dir = root_path / "train" / "unlabel_images" / "images"

    images = _list_files(image_dir)
    labels = _list_files(label_dir)
    unlabeled = _list_files(unlabeled_dir) if unlabeled_dir.exists() else []

    if len(images) != len(labels):
        raise ValueError(
            "The number of training images and labels does not match:"
            f" {len(images)} vs {len(labels)}"
        )

    return images, labels, unlabeled


def collect_validation_paths(root: PathLike) -> Tuple[List[Path], List[Path]]:
    root_path = Path(root)
    image_dir = root_path / "images_cut"
    label_dir = root_path / "labels_cut"

    images = _list_files(image_dir)
    labels = _list_files(label_dir)

    if len(images) != len(labels):
        raise ValueError(
            "The number of validation images and labels does not match:"
            f" {len(images)} vs {len(labels)}"
        )

    return images, labels
