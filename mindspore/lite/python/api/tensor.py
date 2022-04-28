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
Tensor API.
"""
from .lib import _c_lite_wrapper


class Tensor:
    """
    Tensor Class

    Args:
    """
    def __init__(self, tensor):
        if not isinstance(tensor, _c_lite_wrapper.TensorBind):
            raise ValueError(f"Parameter 'tensor' should instance of TensorBind, but actually {type(tensor)}")
        self._tensor = tensor

    @classmethod
    def create_tensor(class_, tensor_name, date_type, shape, data, data_len):
        self._tensor = _c_lite_wrapper.TensorBind(tensor_name, date_type, shape, data, data_len)

    def set_tensor_name(self, tensor_name):
        self._tensor.set_tensor_name(tensor_name)

    def get_tensor_name(self):
        return self._tensor.get_tensor_name()

    def set_data_type(self, data_type):
        self._tensor.set_data_type(data_type)

    def get_data_type(self):
        return self._tensor.get_data_type()

    def set_shape(self, shape):
        self._tensor.set_shape(shape)

    def get_shape(self):
        return self._tensor.get_shape()

    def set_format(self, tensor_format):
        self._tensor.set_format(tensor_format)

    def get_format(self):
        return self._tensor.get_format()

    def get_element_num(self):
        return self._tensor.get_element_num()

    def get_data_size(self):
        return self._tensor.get_data_size()

    def set_data_from_numpy(self, numpy_obj):
        if numpy_obj.nbytes != self.get_data_size():
            raise f"Data size not equal! Numpy size: {numpy_obj.nbytes}, Tensor size: {self.get_data_size()}"
        self._tensor.set_data_from_numpy(numpy_obj)
        self._numpy_obj = numpy_obj  # keep reference count of numpy objects

    def get_data_to_numpy(self):
        return self._tensor.get_data_to_numpy()

    def __str__(self):
        res = f"tensor_name: {self.get_tensor_name()}, " \
              f"data_type: {self.get_data_type()}, " \
              f"shape: {self.get_shape()}, " \
              f"format: {self.get_format()}, " \
              f"element_num, {self.get_element_num()}, " \
              f"data_size, {self.get_data_size()}."
        return res
