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
"""Collate module."""

import copy
import re
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from typing import Any, Callable, Optional, Union

import numpy as np

from mindspore.common import Tensor, float64

# S: bytes string type (bytes).
# a: old version alias for bytes string type (same as S).
# U: Unicode string type.
# O: Python object type.
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def convert_tensor(data: Tensor) -> Tensor:
    """Default convert function for :class:`mindspore.Tensor`."""
    return data


def convert_numpy_scalar(data: Union[np.number, np.bool_]) -> Tensor:
    """Default convert function for :class:`numpy.number` and :class:`numpy.bool_`."""
    return Tensor(data)


def convert_numpy_array(data: np.ndarray) -> Union[Tensor, np.ndarray]:
    """Default convert function for :class:`numpy.ndarray`."""
    # NumPy arrays with dtype `str`, `bytes` or `object` cannot be converted to Tensor.
    if np_str_obj_array_pattern.search(data.dtype.str) is not None:
        return data
    return Tensor.from_numpy(data)


def convert_mapping(data: Mapping) -> Mapping:
    """Default convert function for :class:`collections.abc.Mapping`."""
    try:
        if isinstance(data, MutableMapping):
            # To preserve the original Mapping structure and its properties,
            # shallow copy and update the values in-place.
            clone = copy.copy(data)
            clone.update({key: default_convert(data[key]) for key in data})
            return clone
        return type(data)({key: default_convert(data[key]) for key in data})
    except TypeError:
        # If creating the original Mapping type fails,
        # fall back to creating a dictionary.
        return {key: default_convert(data[key]) for key in data}


def convert_namedtuple(data: tuple) -> tuple:
    """Default convert function for :class:`collections.namedtuple`."""
    return type(data)(*(default_convert(d) for d in data))


def convert_tuple(data: tuple) -> list:
    """Default convert function for :class:`tuple`."""
    # Convert to list for backwards compatibility.
    return [default_convert(d) for d in data]


def convert_sequence(data: Sequence) -> Sequence:
    """Default convert function for :class:`collections.abc.Sequence` except for :class:`str` and :class:`bytes`."""
    try:
        if isinstance(data, MutableSequence):
            # To preserve the original Sequence structure and its properties,
            # shallow copy and update the values in-place.
            clone = copy.copy(data)
            for i, d in enumerate(data):
                clone[i] = default_convert(d)
            return clone
        return type(data)([default_convert(d) for d in data])
    except TypeError:
        # If creating the original Sequence type fails,
        # fall back to creating a list.
        return [default_convert(d) for d in data]


def default_convert(data: Any) -> Any:
    """
    Default function for converting each NumPy array element into a :class:`mindspore.Tensor`
    when batching is disabled in :class:`~mindspore.dataset.dataloader.DataLoader`.

    * If the input is a NumPy array and its dtype is not `str`, `bytes` or `object`, convert it into
      a :class:`mindspore.Tensor`;
    * If the input is a NumPy numeric or boolean scalar, convert it into a :class:`mindspore.Tensor`;
    * If the input is a :py:class:`~collections.abc.Mapping`, keep all the keys unchanged and convert
      the value of each key by calling this function recursively;
    * If the input is a :py:class:`~collections.abc.Sequence`, convert the element at each position
      by calling this function recursively;
    * Otherwise, leave it unchanged.

    Args:
        data (:py:class:`~typing.Any`): A single data to be converted.

    Returns:
        :py:class:`~typing.Any`, the converted data.

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset.dataloader import default_convert
        >>>
        >>> default_convert(np.array([0, 1, 2]))
        Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])
        >>>
        >>> default_convert(np.int32(0))
        Tensor(shape=[], dtype=Int32, value= 0)
        >>>
        >>> default_convert({"data": np.array([0, 1, 2])})
        {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
        >>>
        >>> default_convert([np.array([0, 1, 2]), np.array([3, 4, 5])])
        [Tensor(shape=[3], dtype=Int64, value= [0, 1, 2]), Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])]
        >>>
        >>> default_convert(np.array(["text"]))
        array(['text'], dtype='<U4')
    """
    if isinstance(data, Tensor):
        return convert_tensor(data)
    if isinstance(data, (np.number, np.bool_)):
        return convert_numpy_scalar(data)
    if isinstance(data, np.ndarray):
        return convert_numpy_array(data)
    if isinstance(data, Mapping):
        return convert_mapping(data)
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return convert_namedtuple(data)
    if isinstance(data, tuple):
        return convert_tuple(data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return convert_sequence(data)
    return data


def collate_mapping_fn(
    batch: list[Mapping],
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
) -> Mapping:
    """Collate function for :class:`collections.abc.Mapping`."""
    elem = batch[0]

    try:
        if isinstance(elem, MutableMapping):
            # To preserve the original Mapping structure and its properties,
            # shallow copy and update the values in-place.
            clone = copy.copy(elem)
            clone.update({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
            return clone
        return type(elem)({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
    except TypeError:
        # If creating the original Mapping type fails,
        # fall back to creating a dictionary.
        return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}


def collate_namedtuple_fn(
    batch: list[tuple], *, collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None
) -> tuple:
    """Collate function for :class:`collections.namedtuple`."""
    elem = batch[0]
    return type(elem)(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))


def collate_sequence_fn(
    batch: list[Sequence],
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
) -> Sequence:
    """Collate function for :class:`collections.abc.Sequence`."""
    elem = batch[0]

    if len({len(elem) for elem in batch}) > 1:
        raise RuntimeError("Each element in list of batch must be of equal size.")

    # Group elements at the same position from all samples together.
    # For example, [[1,2], [3,4], [5,6]] -> [(1,3,5), (2,4,6)].
    transposed = list(zip(*batch))

    if isinstance(elem, tuple):
        # Convert to list for backwards compatibility.
        return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    try:
        if isinstance(elem, MutableSequence):
            # To preserve the original Sequence structure and its properties,
            # shallow copy and update the values in-place.
            clone = copy.copy(elem)
            for i, samples in enumerate(transposed):
                clone[i] = collate(samples, collate_fn_map=collate_fn_map)
            return clone
        return type(elem)([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
    except TypeError:
        # If creating the original Sequence type fails,
        # fall back to creating a list.
        return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]


def collate(
    batch: list,
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
) -> Any:
    """
    Collate the input batch of data by the appropriate function for each element type selected from the type
    to collate function mapping defined in `collate_fn_map`.

    All the elements in the batch should be of the same type.

    * If the element type is in `collate_fn_map` or the element is a subclass of the type in `collate_fn_map`,
      use the corresponding collate function to collate the batch;
    * If the element is a :py:class:`~collections.abc.Mapping`, collate by key: for each key, collect the values
      corresponding to that key from all mappings in the batch to form a new batch, recursively call this function
      on that batch, and use the result as the new value for that key. All mappings in the batch must have the same
      keys, and the types of values corresponding to each key must be the same;
    * If the element is a :py:class:`~collections.abc.Sequence`, collate by position: for each position, collect the
      elements at that position from all sequences in the batch to form a new batch, recursively call this function
      on that batch, and use the result as the new element at that position. All sequences in the batch must have the
      same length;
    * Otherwise, raise an exception to indicate that the element type is not supported.

    Each collate function requires a positional argument for `batch` and a keyword argument for `collate_fn_map`.

    Args:
        batch (list): A batch of data to be collated.

    Keyword Args:
        collate_fn_map (Optional[dict[Union[type, tuple[type, ...]], Callable]]): Mapping from element type
            to the corresponding collate function. Default: ``None`` .

    Returns:
        :py:class:`~typing.Any`, the collated data.

    Examples:
        >>> import mindspore
        >>> from mindspore.dataset.dataloader._utils.collate import collate
        >>>
        >>> def collate_int_fn(batch, *, collate_fn_map):
        ...     return mindspore.tensor(batch)
        >>>
        >>> collate_map = {int: collate_int_fn}
        >>>
        >>> collate([0, 1, 2], collate_fn_map=collate_map)
        Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])
        >>>
        >>> collate([{"data": 0, "label": 2}, {"data": 1, "label": 3}], collate_fn_map=collate_map)
        {'data': Tensor(shape=[2], dtype=Int64, value= [0, 1]), 'label': Tensor(shape=[2], dtype=Int64, value= [2, 3])}
        >>>
        >>> collate([(0, 3), (1, 4), (2, 5)], collate_fn_map=collate_map)
        [Tensor(shape=[3], dtype=Int64, value= [0, 1, 2]), Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])]
    """
    elem = batch[0]
    elem_type = type(elem)

    # For types in collate_fn_map, call the corresponding collate function.
    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    # For collection types, call collate function recursively.
    if isinstance(elem, Mapping):
        return collate_mapping_fn(batch, collate_fn_map=collate_fn_map)
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return collate_namedtuple_fn(batch, collate_fn_map=collate_fn_map)
    if isinstance(elem, Sequence):
        return collate_sequence_fn(batch, collate_fn_map=collate_fn_map)

    raise TypeError(f"Cannot find the appropriate collate function for type {type(elem)} in collate_fn_map.")


# pylint: disable=unused-argument
def collate_tensor_fn(
    batch: list[Tensor],
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
) -> Tensor:
    """Collate function for :class:`mindspore.Tensor`."""
    return Tensor(np.stack(batch, axis=0))


def collate_numpy_array_fn(
    batch: list[np.ndarray], *, collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None
) -> Tensor:
    """Collate function for :class:`numpy.ndarray`."""
    elem = batch[0]

    # NumPy arrays with dtype `str`, `bytes` or `object` cannot be converted to Tensor.
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(f"NumPy arrays with dtype {elem.dtype} are not supported for collation.")
    return collate([Tensor.from_numpy(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(
    batch: list[Union[np.bool_, np.number, np.object_]],
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
) -> Tensor:
    """Collate function for :class:`numpy.number`, :class:`numpy.bool_` and :class:`numpy.object_`."""
    return Tensor(batch)


def collate_float_fn(
    batch: list[float], *, collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None
) -> Tensor:
    """Collate function for :class:`float`."""
    return Tensor(batch, dtype=float64)


def collate_int_fn(
    batch: list[int], *, collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None
) -> Tensor:
    """Collate function for :class:`int`."""
    return Tensor(batch)


def collate_str_fn(
    batch: list[Union[str, bytes]], *, collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None
) -> list[Union[str, bytes]]:
    """Collate function for :class:`str` and :class:`bytes`."""
    return batch


default_collate_fn_map: dict[Union[type, tuple[type, ...]], Callable] = {
    Tensor: collate_tensor_fn,
    np.ndarray: collate_numpy_array_fn,
    (np.bool_, np.number, np.object_): collate_numpy_scalar_fn,
    float: collate_float_fn,
    int: collate_int_fn,
    str: collate_str_fn,
    bytes: collate_str_fn,
}


def default_collate(batch: list) -> Any:
    """
    Default function for concatenating a batch of data along the first dimension when batching is enabled
    in :class:`~mindspore.dataset.dataloader.DataLoader`.

    This function uses a predefined mapping from data types to their corresponding collate functions to
    do the following type transformations, then collates the input batch according to the rules described
    in :func:`~mindspore.dataset.dataloader._utils.collate.collate`:

    * :py:class:`list` [:class:`mindspore.Tensor`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:class:`numpy.ndarray`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:py:class:`float`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:py:class:`int`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:py:class:`str`] -> :py:class:`list` [:py:class:`str`]
    * :py:class:`list` [:py:class:`bytes`] -> :py:class:`list` [:py:class:`bytes`]

    Args:
        batch (list): A batch of data to be collated.

    Returns:
        :py:class:`~typing.Any`, the collated data.

    Examples:
        >>> from mindspore.dataset.dataloader import default_collate
        >>>
        >>> default_collate([0, 1, 2])
        Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])
        >>>
        >>> default_collate([{"data": 0, "label": 2}, {"data": 1, "label": 3}])
        {'data': Tensor(shape=[2], dtype=Int64, value= [0, 1]), 'label': Tensor(shape=[2], dtype=Int64, value= [2, 3])}
        >>>
        >>> default_collate([(0, 3), (1, 4), (2, 5)])
        [Tensor(shape=[3], dtype=Int64, value= [0, 1, 2]), Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])]
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)
