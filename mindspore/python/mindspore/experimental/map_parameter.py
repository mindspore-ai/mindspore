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
"""MapParameter implementation."""
from __future__ import absolute_import

__all__ = ['MapParameter']

import numbers
import mindspore as ms
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore._c_expression import Tensor as Tensor_
from mindspore._c_expression import MapTensor_


class MapParameter(Parameter):
    """
    MapParameter is a parameter that stores a map like data structure.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        key_dtype (:class:`mindspore.dtype`): The data type of the key. The argument should be defined in
            `mindspore.dtype`, currently only integer types are supported. Default: int32.
        value_dtype (:class:`mindspore.dtype`): The data type of the value Tensor. The argument should
            be defined in `mindspore.dtype`. Default: float32.
        value_shape (Union[tuple, list, int]): Used to indicate the shape of the value Tensor. The argument should be
            a list of integers, a tuple of integers or an integer. Default: 1.
        default_value (Union[Tensor, str]): The default value Tensor or initializer name. Default: 'zeros'.
        name (str): Name of the map parameter. Default: None.
        requires_grad (bool): True if the parameter requires gradient. Default: True.


    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.experimental import MapParameter
        >>>
        >>> m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3), default_value='zeros')
        >>> t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        >>> m.put(Tensor([1, 2], dtype=ms.int32), Tensor([[1, 1, 1], [2, 2, 2]], dtype=np.float32))
        >>> t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
        >>> print(t)
        [[1. 1. 1.]
         [2. 2. 2.]
         [0. 0. 0.]]
        >>> m.del(Tensor([2, 3], dtype=ms.int32))
        >>> t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
        >>> print(t)
        [[1. 1. 1.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """

    def __new__(cls, key_dtype=ms.int32, value_dtype=ms.float32, value_shape=1, default_value='zeros', **kwargs):
        if isinstance(value_shape, numbers.Number):
            value_shape = (value_shape,)
        data = Tensor_(value_dtype, value_shape)
        obj = Tensor_.__new__(cls)
        Tensor_.__init__(obj, data)
        obj.key_dtype = key_dtype
        obj.value_dtype = value_dtype
        obj.value_shape = value_shape
        obj.default_value = default_value
        return obj

    def __init__(self, name=None, requires_grad=True, **kwargs):
        Parameter.__init__(self, self, name=name, requires_grad=requires_grad)
        self.map_tensor_ = MapTensor_(self.key_dtype, self.value_dtype, self.value_shape)

    def get(self, key_tensor, default_value=None):
        """
        Get value tensor according the key tensor, fill and return the default value if key is not existed.

        Args:
            key_tensor (Tensor): The key tensor.
            default_value (Union[Tensor, str]): The default value or initializer. Default: None

        Returns:
            Tensor, the value tensor for the key tensor.
        """
        if default_value is None:
            default_value = self.default_value
        result = initializer(default_value, shape=(key_tensor.shape + self.value_shape), dtype=self.value_dtype)
        return result.init_data()

    def put(self, key_tensor, value_tensor):
        """
        Insert or update records according the given key tensor and value tensor.

        Args:
            key_tensor (Tensor): The key tensor.
            value_tensor (Tensor): The value tensor.

        Returns:
            MapParameter, the MapParameter object itself.
        """
        return self

    def erase(self, key_tensor):
        """
        Remove records according the given key tensor.

        Args:
            key_tensor (Tensor): The key tensor.

        Returns:
            MapParameter, the MapParameter object itself.
        """
        return self
