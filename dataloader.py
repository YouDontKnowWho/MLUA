"""Reusable batch samplers for the MLUA training scripts."""

from __future__ import annotations

import itertools
from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


def iterate_once(indices: Sequence[int]) -> np.ndarray:
    return np.random.permutation(indices)


def iterate_eternally(indices: Sequence[int]) -> Iterator[np.ndarray]:
    def infinite_shuffles() -> Iterator[np.ndarray]:
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable: Sequence[int], n: int) -> Iterator[tuple[int, ...]]:
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler[List[int]]):
    """Sampler that mixes labelled and unlabelled samples each batch."""

    def __init__(self, l_indices: Sequence[int], ul_indices: Sequence[int], batch_size: int, l_batch_size: int) -> None:
        if not 0 < l_batch_size <= batch_size:
            raise ValueError("l_batch_size must be positive and no larger than batch_size")

        self.l_indices = list(l_indices)
        self.ul_indices = list(ul_indices)
        self.l_batch_size = l_batch_size
        self.ul_batch_size = batch_size - l_batch_size

        if len(self.l_indices) < self.l_batch_size:
            raise ValueError("Not enough labelled samples to fill a batch")
        if self.ul_batch_size > 0 and len(self.ul_indices) < self.ul_batch_size:
            raise ValueError("Not enough unlabelled samples to fill a batch")

    def __iter__(self) -> Iterator[List[int]]:
        label_iter = iterate_once(self.l_indices)
        label_batches = grouper(label_iter, self.l_batch_size)

        if self.ul_batch_size == 0:
            for labeled in label_batches:
                yield list(labeled)
            return

        unlabel_iter = iterate_eternally(self.ul_indices)
        unlabel_batches = grouper(unlabel_iter, self.ul_batch_size)

        for labeled, unlabeled in zip(label_batches, unlabel_batches):
            batch = list(labeled)
            batch.extend(unlabeled)
            yield batch

    def __len__(self) -> int:
        return len(self.l_indices) // self.l_batch_size
