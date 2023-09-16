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

import os
import sys
from copy import copy
import numbers
import mindspore as ms
from mindspore.common.parameter import Parameter, _get_unique_parameter_key
from mindspore._c_expression import Tensor as Tensor_
from mindspore._c_expression import MapTensor_
from mindspore.ops.operations import _map_tensor_ops


class MapParameter(Parameter):
    """
    MapParameter is a parameter that stores a map like data structure.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        key_dtype (:class:`mindspore.dtype`): The data type of the key. The argument should be defined in
            `mindspore.dtype`, currently only integer types are supported. Default: int32.
        value_dtype (:class:`mindspore.dtype`): The data type of the value Tensor. The argument should
            be defined in `mindspore.dtype`. Default: float32.
        value_shape (Union[tuple, list, int]): Used to indicate the shape of the value Tensor. The argument should be
            a list of integers, a tuple of integers or an integer. Default: 1.
        key_tensor (:class:`mindspore.tensor`): The key Tensor.
        value_tensor (:class:`mindspore.tensor`): The value Tensor.
        default_value (Union[numbers.Number, str]): The default value number or initializer name. Default: 'normal'.
        permit_filter_value (numbers.Number): The permit filter value number. Default: 1.
        evict_filter_value (numbers.Number): The evict filter value number. Default: MAX_SIZE.
        name (str): Name of the map parameter. Default: ``None``.
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
        >>> m.put(Tensor([1, 2], dtype=ms.int32), Tensor([[1, 1, 1], [2, 2, 2]], dtype=ms.float32))
        >>> t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
        >>> print(t)
        [[1. 1. 1.]
         [2. 2. 2.]
         [0. 0. 0.]]
        >>> m.erase(Tensor([2, 3], dtype=ms.int32))
        >>> print(t)
        [[1. 1. 1.]]

    """

    def __new__(cls, key_dtype=None, value_dtype=None, value_shape=None, key_tensor=None, value_tensor=None,
                default_value='normal', permit_filter_value=1, evict_filter_value=sys.maxsize, **kwargs):
        if value_dtype is not None:
            if isinstance(value_shape, numbers.Number):
                value_shape = (value_shape,)
            data = Tensor_(value_dtype, value_shape)
        elif value_tensor is not None:
            data = Tensor_(value_tensor.dtype, value_tensor.shape)
        else:
            # default
            data = Tensor_(ms.float32, (1,))
        obj = Tensor_.__new__(cls)
        Tensor_.__init__(obj, data)
        # Compatible attributes with Parameter.
        obj.has_init = False
        obj.init_mode = None
        obj.is_default_input_init = False
        # MapParameter added attributes.
        MapParameter._check_map_parameter_args(key_tensor, key_dtype, value_tensor, value_dtype, value_shape)
        if key_tensor is not None:
            obj.key_dtype = key_tensor.dtype
        else:
            obj.key_dtype = key_dtype if key_dtype is not None else ms.int32

        if value_tensor is not None:
            obj.value_dtype = value_tensor.dtype
        else:
            obj.value_dtype = value_dtype if value_dtype is not None else ms.float32

        if value_tensor is not None:
            obj.value_shape = value_tensor.shape
        else:
            obj.value_shape = value_shape if value_shape is not None else (1,)

        obj.default_value = default_value
        obj.permit_filter_value = permit_filter_value
        obj.evict_filter_value = evict_filter_value
        obj.key_tensor = key_tensor
        obj.value_tensor = value_tensor
        return obj

    def __init__(self, name=None, requires_grad=True, **kwargs):
        Parameter.__init__(self, self, name=name, requires_grad=requires_grad)
        if self.key_tensor is not None and self.value_tensor is not None:
            self._map_tensor = MapTensor_(self.key_tensor, self.value_tensor, self.default_value,
                                          self.permit_filter_value, self.evict_filter_value)
        else:
            self._map_tensor = MapTensor_(self.key_dtype, self.value_dtype, self.value_shape, self.default_value,
                                          self.permit_filter_value, self.evict_filter_value)
        self.map_put = _map_tensor_ops.put
        self.map_erase = _map_tensor_ops.erase

    def __getitem__(self, key_tensor):
        return self.get(key_tensor, True)

    def __setitem__(self, key_tensor, value_tensor):
        return self.put(key_tensor, value_tensor)

    def __str__(self):
        return 'MapParameter(' + str(self._map_tensor) + ')'

    def __copy__(self):
        x = type(self)()
        x.__dict__.update(self.__dict__)
        return x

    @staticmethod
    def _check_map_parameter_args(key_tensor, key_dtype, value_tensor, value_dtype, value_shape):
        if key_dtype is not None and key_tensor is not None and key_dtype != key_tensor.dtype:
            raise ValueError(f"When initializing a MapParameter, 'key_dtype' and 'key_tensor.dtype' should be set the"
                             f" same.")
        if value_dtype is not None and value_tensor is not None and value_dtype != value_tensor.dtype:
            raise ValueError(f"When initializing a MapParameter, 'value_dtype' and 'value_tensor.dtype' should be set "
                             f"the same.")
        if value_shape is not None and value_tensor is not None and value_shape != value_tensor.shape:
            raise ValueError(f"When initializing a map_parameter, 'value_shape' and 'value_tensor.shape' should be set "
                             f"the same.")

    def clone(self, init='same'):
        """
        Clone the MapParameter.

        Args:
            init (Union[str, numbers.Number]): Initialize the default value of the new map parameter.
                If `init` is a `numbers.Number`, clone a new map parameter with the same key value shape
                and dtype, and the default value of the new map parameter will be set according to `init`.
                If `init` is a `str`, the `init` should be the alias of the class inheriting from `Initializer`.
                If `init` is 'same', clone a new map parameter with the same default value. Default: 'same'.

        Returns:
            MapParameter, the new map parameter.
        """
        x = copy(self)
        x.param_info = self.param_info.clone()
        info = self.param_info
        if hasattr(info, "cloned_obj"):
            info.cloned_obj.append(x)
        else:
            info.cloned_obj = [x]
        self.param_info = info
        if init != 'same':
            x.default_value = init
        x._map_tensor = MapTensor_(x.key_dtype, x.value_dtype, x.value_shape, x.default_value, x.permit_filter_value,
                                   x.evict_filter_value)
        x.cache_enable = self.cache_enable
        if x.cache_enable:
            x.key = _get_unique_parameter_key()
        return x

    def get(self, key_tensor, insert_default_value=True):
        """
        Get value tensor according the key tensor, fill and return the default value in map parameter if key is not
        existed.

        Args:
            key_tensor (Tensor): The key tensor.
            insert_default_value (bool): The flag of insert default_value.

        Returns:
            Tensor, the value tensor for the key tensor.
        """
        map_get = _map_tensor_ops.MapTensorGet(insert_default_value)
        return map_get(self._map_tensor, key_tensor)

    def get_keys(self):
        """
        Get all keys as a tensor.

        Returns:
            Tensor, the tensor contains all keys.
        """
        return self._map_tensor.get_keys()

    def get_values(self):
        """
        Get all values as a tensor.

        Returns:
            Tensor, the tensor contains all values.
        """
        return self._map_tensor.get_values()

    def get_data(self):
        """
        Get all keys and values as a tensor.

        Returns:
            Tensor, the tensor contains all keys and values.
        """
        return self._map_tensor.get_data()

    def put(self, key_tensor, value_tensor):
        """
        Insert or update records according the given key tensor and value tensor.

        Args:
            key_tensor (Tensor): The key tensor.
            value_tensor (Tensor): The value tensor.

        Returns:
            MapParameter, the MapParameter object itself.
        """
        self.map_put(self._map_tensor, key_tensor, value_tensor)
        return self._map_tensor

    def erase(self, key_tensor):
        """
        Remove records according the given key tensor.

        Args:
            key_tensor (Tensor): The key tensor.

        Returns:
            MapParameter, the MapParameter object itself.
        """
        self.map_erase(self._map_tensor, key_tensor)
        return self._map_tensor

    def export_data(self, incremental=False):
        """
        Export data from this map parameter.

        Args:
            incremental (bool): False for full export, otherwise for incremental export. Default: False.
            When exporting data incrementally, the value_array does not contain unchanged data.The length
            of the key_array and the length of the status_array are consistent.

        Returns:
            Tuple(key_array, value_array, status_array), The exported data as a tuple.
        """
        return self._map_tensor.export_data(incremental)

    def export_bytes(self, incremental=False):
        """
        Export bytes from this map parameter.

        Args:
            incremental (bool): False for full export, otherwise for incremental export. Default: False.
            When exporting data incrementally, the value_array does not contain unchanged data. The length
            of the key_array and the length of the status_array are consistent.

        Returns:
            Tuple(bytes, bytes, bytes), The exported bytes as a tuple.
        """
        return self._map_tensor.export_bytes(incremental)

    def import_data(self, data):
        """
        Import this map parameter from exported data.

        Args:
            data (Tuple): The data tuple with key_array, value_array and status_array.
        """
        self._map_tensor.import_data(data)

    def export_slice_data(self, incremental=False):
        """
        Export a slice data from this map parameter.
        When MapParameter occupies a large memory, only one slice
        of MapParameter is exported at a time (the default slice size is 1GB).

        Args:
            incremental (bool): False for full export, otherwise for incremental export. Default: False.
            When exporting data incrementally, the value_array does not contain unchanged data.The length
            of the key_array and the length of the status_array are consistent.

        Returns:
            Tuple(key_array, value_array, status_array, last_slice), The exported data as a tuple, and
            the last_slice is bool variable and means whether finish export.
        """
        enable_persistent = "MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE" in os.environ
        if not enable_persistent:
            return self._map_tensor.export_slice_data(incremental)
        return self._map_tensor.export_persistent_slice_data(self.key, incremental)
