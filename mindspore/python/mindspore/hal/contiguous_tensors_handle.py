# This is the Python alloc contiguous memory handle.
#
# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Contiguous memory handle."""
from mindspore.common.tensor import Tensor
from mindspore.graph.api import _convert_python_data
from mindspore.common.dtype import type_size_in_bytes
from mindspore._c_expression import slice_by_tensor_index, slice_by_padding_shape, \
    combine_tensor_list_contiguous as combine_tensor_list, TensorPy as Tensor_


def combine_tensor_list_contiguous(tensor_list, enable_mem_align=True):
    r"""
    Return a contiguous memory handle where contiguous memory has been requested and slicing functionality is provided.

    Args:
        tensor_list (list[Tensor], tuple[Tensor]): The tensor list to be stored.
        enable_mem_align (bool, optional): Whether to enable the memory alignment function.
            False is not supported. Default ``True`` .

    Returns:
            ContiguousTensorsHandle, a manager with contiguous memory.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.hal.contiguous_tensors_handle import combine_tensor_list_contiguous
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> handle = combine_tensor_list_contiguous([x, y], True)
        >>> print(handle[0].shape)
        [1]
        >>> print(handle[1: 3].asnumpy())
        [2, 3]
        >>> print(output.slice_by_tensor_index(0, 1).asnumpy())
        [1, 2, 3]
    """
    return ContiguousTensorsHandle(tensor_list, enable_mem_align)


class ContiguousTensorsHandle:
    r"""
    ContiguousTensorsHandle is a handle manage continuous memory.

    Args:
        tensor_list (list[Tensor], tuple[Tensor]): The tensor list to be stored.
        enable_mem_align (bool, optional): Whether to enable the memory alignment function.
            False is not supported. Default ``True`` .

    Returns:
        ContiguousTensorsHandle, a manager with contiguous memory.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.hal.contiguous_tensors_handle import ContiguousTensorsHandle
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> handle = ContiguousTensorsHandle([x, y], True)
        >>> print(handle[0].shape)
        [1]
        >>> print(handle[1: 3].asnumpy())
        [2, 3]
    """

    def __init__(self, tensor_list, enable_mem_align=True):
        if isinstance(tensor_list, (list, tuple)):
            for tensor in tensor_list:
                if not isinstance(tensor, (Tensor, Tensor_)):
                    raise TypeError(f"input list must be [Tensor, ...].")
            if isinstance(tensor_list, list):
                self.tensor_list = tuple(tensor_list)
            else:
                self.tensor_list = tensor_list
        else:
            raise TypeError(f"input list must be [Tensor, ...].")
        if not isinstance(enable_mem_align, bool):
            raise TypeError(f"enable_mem_align must be bool.")
        padding_sizes_pair = combine_tensor_list(self.tensor_list, enable_mem_align)
        self.before_padding_sizes = padding_sizes_pair[0]
        self.after_padding_sizes = padding_sizes_pair[1]
        self.total_padding_size = sum(self.after_padding_sizes)
        self.handle_shape = self.total_padding_size / type_size_in_bytes(self.tensor_list[0].dtype)
        self.enable_mem_align = enable_mem_align

    def __getitem__(self, item):
        """
        item is sliced by shape
        :param item:
        :return: Tensor
        """
        start = 0
        end = int(self.handle_shape)
        if isinstance(item, slice):
            if item.start is not None:
                start = item.start
            if item.stop is not None:
                end = item.stop
            if not isinstance(start, int) or not isinstance(end, int):
                raise TypeError(f"slice input error.")
            if start < 0 or end > self.handle_shape or start >= end:
                raise ValueError(f"slice input error.")
            return _convert_python_data(slice_by_padding_shape(self.tensor_list[0], start, end))
        if not isinstance(item, int):
            raise TypeError(f"slice input must be "
                            f"1.index -> int."
                            f"2.[start: end: step] -> [int: int: int].")
        if item < 0 or item > self.handle_shape:
            raise ValueError(f"slice input is out of tensor_list size.")
        return _convert_python_data(slice_by_padding_shape(self.tensor_list[0], item, item + 1))

    def __str__(self):
        list_str = "Handle total size: " + str(self.total_padding_size) + "\n"
        index = 0
        for tensor in self.tensor_list:
            list_str = list_str + "Tensor[" + str(index) + "]: " + str(tensor.asnumpy()) + "\n"
            index += 1
        return list_str

    def slice_by_tensor_index(self, start=None, end=None):
        """
        Return the tensor which is sliced by tensor index.

        Args:
            start(int, None): Starting position. Default ``None``.
            end(int, None): Deadline position. Default ``None``.

        Returns:
            Tensor

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> from mindspore.hal.contiguous_tensors_handle import ContiguousTensorsHandle
            >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
            >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
            >>> handle = ContiguousTensorsHandle([x, y], True)
            >>> print(output.slice_by_tensor_index(0, 1).asnumpy())
            [1, 2, 3]
        """
        index_start = 0
        index_end = len(self.tensor_list)
        if start is not None:
            index_start = start
            if end is None:
                index_end = index_start + 1
        if end is not None:
            index_end = end
        if not isinstance(index_start, int) or not isinstance(index_end, int):
            raise TypeError(f"slice input error.")

        if index_start < 0 or index_end > len(self.tensor_list) or index_start >= index_end:
            raise ValueError(f"slice input error.")
        return _convert_python_data(slice_by_tensor_index(self.tensor_list, self.before_padding_sizes,
                                                          self.after_padding_sizes, index_start, index_end))
