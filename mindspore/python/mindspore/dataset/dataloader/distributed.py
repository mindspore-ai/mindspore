# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Distributed sampler module."""

import math
from collections.abc import Iterator
from typing import Optional, TypeVar

import numpy as np

import mindspore as ms
from .dataset import Dataset
from .sampler import Sampler

_T_co = TypeVar("_T_co", covariant=True)


class DistributedSampler(Sampler[_T_co]):
    """
    A sampler that partitioning datasets for distributed training.

    Args:
        dataset (Dataset): Dataset used for sampling.
        num_replicas (int, optional): Number of shards participating in distributed training. Default: ``None`` .
        rank (int, optional): The sequence number of the current shard within `num_replicas`. Default: ``None`` .
        shuffle (bool, optional): Whether the sampler shuffle samples randomly. Default: ``True`` .
        seed (int, optional): When `shuffle` is set to `True` , the seed value used for randomizing the sampler.
            Default: ``0`` .
        drop_last (bool, optional): Whether the sampler discards trailing data. If ``True`` ,
            the sampler discards trailing data to enable equal distribution across all shards;
            if ``False`` , the sampler adds extra indices to enable equal distribution across shards.
            Default: ``False`` .

    Examples:
        >>> from mindspore.dataset.dataloader import DistributedSampler
        >>>
        >>> dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sampler = DistributedSampler(dataset, num_replicas=3, rank=0)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False,
    ) -> None:
        super().__init__(dataset)
        if num_replicas is not None and not isinstance(num_replicas, int):
            raise TypeError(f"num_replicas must be int, but got: {type(num_replicas).__name__}")
        if rank is not None and not isinstance(rank, int):
            raise TypeError(f"rank must be int, but got: {type(rank).__name__}")
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be bool, but got: {type(shuffle).__name__}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, but got: {type(seed).__name__}")
        if not isinstance(drop_last, bool):
            raise TypeError(f"drop_last must be bool, but got: {type(drop_last).__name__}")
        if num_replicas is None:
            if not ms.mint.distributed.is_available():
                raise RuntimeError("MindSpore distributed feature is not available.")
            num_replicas = ms.mint.distributed.get_world_size()
        if rank is None:
            if not ms.mint.distributed.is_available():
                raise RuntimeError("MindSpore distributed feature is not available.")
            rank = ms.mint.distributed.get_rank()
        if num_replicas <= 0:
            raise ValueError(f"Invalid num_replicas: {num_replicas}, num_replicas must be greater than 0.")
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank: {rank}, rank must be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if len(self.dataset) % self.num_replicas == 0 or not self.drop_last:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        else:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        self.total_samples = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = np.random.default_rng(self.seed + self.epoch)
            indices = g.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_samples - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self.total_samples]
        if len(indices) != self.total_samples:
            raise RuntimeError("The length of total indices must be equal to the number of total samples.")

        indices = indices[self.rank : self.total_samples : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("The length of indices must be equal to the number of samples in this replica.")

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        Args:
            epoch (int): Epoch number.
        """
        if not isinstance(epoch, int):
            raise TypeError(f"epoch must be int, but got: {type(epoch).__name__}")
        self.epoch = epoch
