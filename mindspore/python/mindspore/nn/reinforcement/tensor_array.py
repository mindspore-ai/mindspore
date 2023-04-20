# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
TensorArray
"""
from __future__ import absolute_import

from mindspore.nn.cell import Cell
from mindspore.ops.operations import _tensor_array as ta
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype


class TensorArray(Cell):
    r"""TensorArray: a dynamic array to store tensors.

    .. warning::
        This is an experiential prototype that is subject to change and/or deletion.

    Args:
        dtype (mindspore.dtype): the data type in the TensorArray.
        element_shape (tuple[int]): the shape of each tensor in a TensorArray.
        dynamic_size (bool): if ``true`` , the size of TensorArray can be increased. Default: ``True`` .
        size (int): if dynamic_size=False, `size` means the max_size of the TensorArray.
        name (string): the name of this TensorArray. Default: ``"TA"`` .

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> ta = nn.TensorArray(mindspore.int64, ())
        >>> ta.write(0, 1)
        >>> ta.write(1, 2)
        >>> ans = ta.read(1)
        >>> print(ans)
        2
        >>> s = ta.stack()
        >>> print(s)
        [1 2]
        >>> ta.clear()
        >>> ta.write(0, 3)
        >>> ans = ta.read(0)
        >>> print(ans)
        3
        >>> ta.close()
    """
    def __init__(self, dtype, element_shape, dynamic_size=True, size=0, name="TA"):
        """Initialize TensorArray"""
        super(TensorArray, self).__init__()
        Validator.check_subclass("dtype", dtype, mstype.number_type + (mstype.bool_,), self.cls_name)
        Validator.check_int(size, 0, Validator.GE, "size", self.cls_name)
        self.handle_ = ta.TensorArray(dtype, element_shape, dynamic_size, size, name)()
        self.tensor_array_write = ta.TensorArrayWrite()
        self.tensor_array_read = ta.TensorArrayRead(dtype, element_shape)
        self.tensor_array_close = ta.TensorArrayClose()
        self.tensor_array_clear = ta.TensorArrayClear()
        self.tensor_array_stack = ta.TensorArrayStack(dtype, element_shape, dynamic_size, size)
        self.tensor_array_size = ta.TensorArraySize()

    def write(self, index, value):
        """
        Write value(Tensor) to TensorArray in position index.

        Args:
            index ([int, mindspore.int64]): The position to write.
            value (Tensor): The value to add into the TensorArray.

        Returns:
            Bool, true.
        """
        self.tensor_array_write(self.handle_, index, value)
        return True

    def read(self, index):
        """
        Read tensor form the TensorArray by the given position index.

        Args:
            index ([int, mindspore.int64]): The given index to get the tensor.

        Returns:
            Tensor, the value in position index.
        """
        value = self.tensor_array_read(self.handle_, index)
        return value

    def close(self):
        """
        Close the created TensorArray.

        .. warning::
            Once close the TensorArray, every functions belong to this TensorArray will be disaviliable.
            Every resources created in TensorArray will be removed. If this TensorArray will be used in next step
            or somewhere, eg: next loop, please use `clear` instead.

        Returns:
            Bool, true.
        """
        self.tensor_array_close(self.handle_)
        return True

    def clear(self):
        """
        Clear the created TensorArray. Only reset the TensorArray, clear the data and reset the size
        in TensorArray and keep the instance of this TensorArray.

        Returns:
            Bool, true.
        """
        self.tensor_array_clear(self.handle_)
        return True

    def stack(self):
        """
        Stack the values in TensorArray into a stacked Tensor.

        Returns:
            Tensor, all the values will be stacked into one tensor.
        """
        ans = self.tensor_array_stack(self.handle_)
        return ans

    def size(self):
        """
        The logical size of TensorArray.

        Returns:
            Tensor, the size of TensorArray.
        """
        size = self.tensor_array_size(self.handle_)
        return size
