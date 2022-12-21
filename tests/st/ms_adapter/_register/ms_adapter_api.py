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
""" MSAdapter api. """

import sys
import mindspore as ms


class Tensor(ms.Tensor):
    def __init__(self, input_data=None, dtype=None, shape=None, init=None, inner=False):
        super(Tensor, self).__init__(input_data=input_data, dtype=dtype, shape=shape, init=init)

    @property
    def attr(self):
        return 10

    def method(self, x):
        return x + self.attr

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]


class Parameter(ms.Parameter):
    def __new__(cls, default_input, *args, **kwargs):
        init_data_flag = bool(isinstance(default_input, ms.Tensor) and default_input.has_init)
        rc = sys.getrefcount(default_input)
        _, *class_init_args = Parameter._get_parameter_new_args(default_input, rc)
        new_type = Parameter._get_base_class(Tensor)
        obj = Tensor.__new__(new_type)
        Tensor.__init__(obj, *class_init_args)
        obj.init_mode = None
        obj.is_default_input_init = init_data_flag
        if obj.has_init:
            obj.init_mode = default_input
        return obj

    @staticmethod
    def _get_base_class(input_class):
        input_class_name = Parameter.__name__
        if input_class_name in Parameter._base_type:
            new_type = Parameter._base_type.get(input_class_name)
        else:
            new_type = type(input_class_name, (Parameter, input_class), {})
            Parameter._base_type[input_class_name] = new_type
        return new_type
