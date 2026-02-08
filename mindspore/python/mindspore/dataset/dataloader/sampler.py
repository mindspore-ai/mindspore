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
"""Sampler module."""

import itertools
from typing import Generic, Iterable, Iterator, TypeVar, Union

import numpy as np

from ._utils import check_type, check_positive

_T_co = TypeVar("_T_co", covariant=True)


class Sampler(Generic[_T_co]):
    """
    Base Class of the Sampler

    Args:
        data_source (Dataset, optional): Dataset to be sampled. Default: ``None`` .
    """

    def __init__(self, data_source=None) -> None:
        pass


class SequentialSampler(Sampler):
    """
    Samples the dataset elements sequentially.

    Args:
        data_source (Dataset): Dataset to be sampled.

    Examples:
        >>> from mindspore.dataset.dataloader import SequentialSampler
        >>>
        >>> dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sampler = SequentialSampler(dataset)
    """

    def __init__(self, data_source) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        yield from range(len(self.data_source))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """
    Samples the dataset elements randomly.

    Args:
        data_source (Dataset): Dataset to be sampled.
        replacement (bool, optional): Whether to enable the return sampling. Default: ``False`` .
        num_samples (Union[int, None], optional): Number of samples to be drawn. Default: ``None`` ,
            will be set to the length of `data_source` .
        generator (numpy.random.Generator, optional): Generator used during sampling. Default: ``None`` .

    Examples:
        >>> from mindspore.dataset.dataloader import RandomSampler
        >>>
        >>> dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sampler = RandomSampler(dataset)
    """

    def __init__(
        self,
        data_source,
        replacement: bool = False,
        num_samples: Union[int, None] = None,
        generator=None,
    ) -> None:
        super().__init__(data_source)
        if not isinstance(replacement, bool):
            raise TypeError(f"replacement must be bool, but got: {type(replacement).__name__}")
        if num_samples is not None and not isinstance(num_samples, int):
            raise TypeError(f"num_samples must be int, but got: {type(num_samples).__name__}")
        if num_samples is not None and num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer value, but got num_samples = {num_samples}")
        if generator is not None and not isinstance(generator, np.random.Generator):
            raise TypeError(f"generator must be numpy.random.Generator, but got: {type(generator).__name__}")
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @property
    def num_samples(self) -> int:
        """
        Get the number of samples to be drawn.

        Returns:
            int, the number of samples to be drawn.
        """

        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = np.random.randint(low=0, high=np.iinfo(np.int64).max + 1, dtype=np.int64)
            generator = np.random.default_rng(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from generator.integers(low=0, high=n, size=(32,), dtype=np.int64).tolist()
            yield from generator.integers(low=0, high=n, size=(self.num_samples % 32,), dtype=np.int64).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from generator.permutation(n).tolist()
            yield from generator.permutation(n).tolist()[: self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[list[int]]):
    """
    Sampler that yields a mini-batch of indices each time.

    Args:
        sampler (Union[Sampler, Iterable]): Sampler used to generate individual indices.
        batch_size (int): Size of the mini-batch.
        drop_last (bool): Whether to drop the last batch if its size is less than `batch_size`.

    Examples:
        >>> from mindspore.dataset.dataloader import BatchSampler, SequentialSampler
        >>>
        >>> dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sequential_sampler = SequentialSampler(dataset)
        >>>
        >>> batch_sampler = BatchSampler(sequential_sampler, 4, False)
        >>> print(list(batch_sampler))
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        >>>
        >>> batch_sampler = BatchSampler(sequential_sampler, 4, True)
        >>> print(list(batch_sampler))
        [[0, 1, 2, 3], [4, 5, 6, 7]]
    """

    def __init__(self, sampler: Union[Sampler, Iterable], batch_size: int, drop_last: bool) -> None:
        super().__init__()
        check_type(batch_size, "batch_size", valid_type=int, invalid_type=bool)
        check_positive(batch_size, "batch_size")
        check_type(drop_last, "drop_last", valid_type=bool)

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            # zip will call elements of args in sequence, equals to call generator batch-size times
            for batch_droplast in zip(*args):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) - 1) // self.batch_size + 1


class InfiniteSampler(Sampler):
    """
    Used as sampler for :class:`~mindspore.dataset.dataloader.IterableDataset`.
    """

    def __iter__(self):
        # pylint: disable=infinite-loop
        while True:
            yield None
