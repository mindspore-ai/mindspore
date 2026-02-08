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
"""Dataset module."""

from typing import Generic, Iterable, Iterator, TypeVar

from mindspore.common import Tensor

_T_co = TypeVar("_T_co", covariant=True)


class Dataset(Generic[_T_co]):
    """
    Base class for implementing all datasets.

    Map style datasets should inherit from it.

    Map style datasets represent a mapping from keys to data samples.
    Subclasses must overwrite :meth:`__getitem__` method, defining how to retrieve the samples according to the key.
    Subclasses could optionally overwrite :meth:`__len__` method, returning the total size of the samples.
    If not implemented, some built-in samplers and :class:`~mindspore.dataset.dataloader.DataLoader` methods may not
    be available.

    Examples:
        >>> from mindspore.dataset.dataloader import Dataset
        >>>
        >>> class MapStyleDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[index]
        ...
        ...     def __len__(self):
        ...         return len(self.data)
    """

    def __init__(self) -> None:
        pass

    def __getitem__(self, index: int) -> _T_co:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __getitem__ method.")


class IterableDataset(Dataset[_T_co], Iterable[_T_co]):
    """
    Base class for implementing iterable datasets.

    Iterable style datasets should inherit from it.

    Iterable style datasets represent an iterable over data samples. It is particularly useful
    when random reads are expensive or even improbable.
    Subclasses must overwrite :meth:`__iter__` method, returning an iterator of samples over the dataset.

    Examples:
        >>> from mindspore.dataset.dataloader import IterableDataset, get_worker_info
        >>>
        >>> class IterableStyleDataset(IterableDataset):
        ...     def __init__(self, num_samples):
        ...         self.start = 0
        ...         self.end = num_samples
        ...
        ...     def __iter__(self):
        ...         worker_info = get_worker_info()
        ...         if worker_info is None:
        ...             return iter(range(self.start, self.end))
        ...         else:
        ...             worker_id = worker_info.id
        ...             num_workers = worker_info.num_workers
        ...             return iter(range(self.start + worker_id, self.end, num_workers))
    """

    def __iter__(self) -> Iterator[_T_co]:
        raise NotImplementedError(f"{self.__class__.__name__} must implement __iter__ method.")


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    """
    Dataset that defined by a collection of :class:`mindspore.Tensor` .

    Each :class:`~mindspore.Tensor` represent a feature column of the dataset, and must have the same size in the
    first dimension, which means the total number of samples.
    Samples will be retrieved by indexing :class:`~mindspore.Tensor` along their first dimension.

    Args:
        *tensors (mindspore.Tensor): A collection of :class:`mindspore.Tensor`.

    Examples:
        >>> from mindspore import Tensor, int32
        >>> from mindspore.dataset.dataloader import TensorDataset
        >>>
        >>> dataset = TensorDataset(Tensor([0, 1], dtype=int32), Tensor([2, 3], dtype=int32))
        >>> for sample in dataset:
        ...     print(sample)
        (Tensor(shape=[], dtype=Int32, value= 0), Tensor(shape=[], dtype=Int32, value= 2))
        (Tensor(shape=[], dtype=Int32, value= 1), Tensor(shape=[], dtype=Int32, value= 3))
    """

    def __init__(self, *tensors: Tensor) -> None:
        super().__init__()
        if any(tensor.shape[0] != tensors[0].shape[0] for tensor in tensors):
            raise ValueError("All tensors must have the same size in the first dimension.")
        self.tensors = tensors

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]
