# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
""" MSAdapter api. """

import sys
import numpy as np
import mindspore as ms
from mindspore import Tensor as ms_Tensor
from mindspore.common import dtype as mstype
from mindspore.common._stub_tensor import StubTensor
from mindspore.common.initializer import Zero
from mindspore._c_expression import Tensor as Tensor_


_dtypeDict = {
    'float16': mstype.float16,
    'float32': mstype.float32,
    'float64': mstype.float64,
    'int8': mstype.int8,
    'int16': mstype.int16,
    'int32': mstype.int32,
    'int64': mstype.int64,
    'uint8': mstype.uint8,
    'bool': mstype.bool_,
    'complex64': mstype.complex64,
    'complex128': mstype.complex128,
    'long': mstype.int64,
    'half': mstype.float16,
    'int': mstype.int32,
    'double': mstype.float64,
    'float': mstype.float32,
    'char': mstype.int8,
    'byte': mstype.uint8,
    'short': mstype.int16
}


class Tensor(StubTensor):
    def __init__(self, *data, dtype=None, inner=False, cast_tensor=False):
        if cast_tensor:
            if len(data) != 1:
                raise RuntimeError("Tensor init data length is not 1 when cast_tensor=True")
            input_data = data[0]
            if isinstance(input_data, StubTensor):
                super(Tensor, self).__init__(stub=input_data.stub, tensor=input_data.tensor)
            elif isinstance(input_data, ms.Tensor):
                super(Tensor, self).__init__(tensor=input_data)
            else:
                raise ValueError(f"Tensor init data type is invalid: {type(input_data)}")
            self.adapter_flag = True
            return

        if dtype is not None:
            key = str(dtype).split('.')[-1].lower()
            dtype = _dtypeDict.get(key)

        if inner is True:
            init_tensor = ms_Tensor(*data, dtype=dtype)
        else:
            _input_data, _shape = self._process_data(data)
            if _shape:
                if dtype is None:
                    dtype = mstype.float32
                init_tensor = ms_Tensor(shape=_shape, dtype=dtype, init=Zero())
                init_tensor.init_data()
            else:
                if dtype is None:
                    if not isinstance(_input_data, (ms.Tensor, Tensor_)):
                        dtype = mstype.float32
                init_tensor = ms_Tensor(input_data=_input_data, dtype=dtype)
        super(Tensor, self).__init__(tensor=init_tensor)
        self.adapter_flag = True

    @property
    def attr(self):
        return 10

    def method(self, x):
        return x + self.attr

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def _process_data(self, data):
        _shape = None
        _input_data = None
        if len(data) == 1:
            if isinstance(data[0], int):
                _shape = data
            elif isinstance(data[0], (np.ndarray, ms.Tensor, list, Tensor_)):
                _input_data = data[0]
            elif isinstance(data[0], tuple):
                if len(data[0]) == 1:
                    _shape = data[0]
                else:
                    _input_data = data[0]
            else:
                raise TypeError(f"For Tensor, data must be a sequence, got {type(data[0])}")
        elif len(data) > 1:
            if not isinstance(data[0], int):
                raise TypeError("For Tensor, elements of shape must be int.")
            _shape = data
        else:
            _input_data = ()
        return _input_data, _shape


class Parameter(ms.Parameter):
    _base_type = {}

    def __new__(cls, data, *args, **kwargs):
        init_data_flag = bool(isinstance(data, ms.Tensor) and data.has_init)
        rc = sys.getrefcount(data)
        input_class, *class_init_args = Parameter._get_parameter_new_args(data, rc)
        new_type = Parameter._get_base_class(input_class)
        obj = input_class.__new__(new_type)
        input_class.__init__(obj, *class_init_args)
        obj.init_mode = None
        obj.is_default_input_init = init_data_flag
        if obj.has_init:
            obj.init_mode = data
        return obj

    def __init__(self, data, name=None, requires_grad=True, layerwise_parallel=False, parallel_optimizer=True):
        self.adapter_flag = True
        super().__init__(default_input=data, name=name, requires_grad=requires_grad,
                         layerwise_parallel=layerwise_parallel, parallel_optimizer=parallel_optimizer)

    @staticmethod
    def _get_base_class(input_class):
        input_class_name = Parameter.__name__
        if input_class_name in Parameter._base_type:
            new_type = Parameter._base_type.get(input_class_name)
        else:
            new_type = type(input_class_name, (Parameter, input_class), {})
            Parameter._base_type[input_class_name] = new_type
        return new_type
