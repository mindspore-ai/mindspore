# Copyright 2022 Huawei Technologies Co., Ltd
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
TensorsQueue, each element in the queue is a list of tensors.
"""
from __future__ import absolute_import

from mindspore.nn.cell import Cell
from mindspore.ops.operations import _rl_inner_ops as rl_ops
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype


class TensorsQueue(Cell):
    r'''
    TensorsQueue: a queue which stores tensors lists.

    .. warning::
        This is an experiential prototype that is subject to change and/or deletion.

    Args:
        dtype (mindspore.dtype): the data type in the TensorsQueue. Each tensor should have the same dtype.
        shapes (tuple[int64]): the shape of each element in TensorsQueue.
        size (int): the size of the TensorsQueue.
        name (string): the name of this TensorsQueue. Default: "TQ".

    Raises:
        TypeError: If `dtype` is not mindpore number type.
        ValueError: If `size` is less than 0.
        ValueError: If `shapes` size is less than 1.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> data1 = Tensor([[0, 1], [1, 2]], dtype=ms.float32)
        >>> data2 = Tensor([1], dtype=ms.float32)
        >>> tq = nn.TensorsQueue(dtype=ms.float32, shapes=((2, 2), (1,)), size=5)
        >>> tq.put((data1, data2))
        >>> ans = tq.pop()
    '''

    def __init__(self, dtype, shapes, size=0, name="TQ"):
        """Initialize TensorsQueue"""
        super(TensorsQueue, self).__init__()
        Validator.check_subclass("dtype", dtype, mstype.number_type + (mstype.bool_,), self.cls_name)
        Validator.check_int(size, 0, Validator.GE, "size", self.cls_name)
        elements_num = len(shapes)
        Validator.check_int(elements_num, 1, Validator.GE, "len(shapes)", self.cls_name)
        self.handle_ = rl_ops.TensorsQueueCreate(dtype, shapes, size, name)()
        self.tensors_q_put = rl_ops.TensorsQueuePut(dtype, shapes)
        self.tensors_q_get = rl_ops.TensorsQueueGet(dtype, shapes)
        self.tensors_q_pop = rl_ops.TensorsQueueGet(dtype, shapes, pop_after_get=True)
        self.tensors_q_clear = rl_ops.TensorsQueueClear()
        self.tensors_q_close = rl_ops.TensorsQueueClose()
        self.tensors_q_size = rl_ops.TensorsQueueSize()

    def put(self, element):
        """
        Put element(tuple(Tensors)) to TensorsQueue in the end of queue.

        Args:
            element (tuple(Tensor) or list[tensor]): The input element.

        Returns:
            Bool, true.
        """
        self.tensors_q_put(self.handle_, element)
        return True

    def get(self):
        """
        Get one element int the front of the TensorsQueue.

        Returns:
            tuple(Tensors), the element in TensorsQueue.
        """
        element = self.tensors_q_get(self.handle_)
        return element

    def pop(self):
        """
        Get one element int the front of the TensorsQueue, and remove it.

        Returns:
            tuple(Tensors), the element in TensorsQueue.
        """
        element = self.tensors_q_pop(self.handle_)
        return element

    def size(self):
        """
        Get the used/available size of the TensorsQueue, and remove it.

        Returns:
            Tensor(mindspore.int64), the used size of TensorsQueue.
        """
        size = self.tensors_q_size(self.handle_)
        return size

    def close(self):
        """
        Close the created TensorsQueue.

        .. warning::
            Once close the TensorsQueue, every functions belong to this TensorsQueue will be disaviliable.
            Every resources created in TensorsQueue will be removed. If this TensorsQueue will be used in next step
            or somewhere, eg: next loop, please use `clear` instead.

        Returns:
            Bool, true.
        """
        self.tensors_q_close(self.handle_)
        return True

    def clear(self):
        """
        Clear the created TensorsQueue. Only reset the TensorsQueue, clear the data and reset the size
        in TensorsQueue and keep the instance of this TensorsQueue.

        Returns:
            Bool, true.
        """
        self.tensors_q_clear(self.handle_)
        return True
