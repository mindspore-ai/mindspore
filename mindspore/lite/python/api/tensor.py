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
from enum import Enum

import numpy

from .lib import _c_lite_wrapper

__all__ = ['DataType', 'Format', 'Tensor']


class DataType(Enum):
    """
    The Enum of data type.
    """
    UNKNOWN = 0
    BOOL = 30
    INT8 = 32
    INT16 = 33
    INT32 = 34
    INT64 = 35
    UINT8 = 37
    UINT16 = 38
    UINT32 = 39
    UINT64 = 40
    FLOAT16 = 42
    FLOAT32 = 43
    FLOAT64 = 44
    INVALID = 2147483647  # INT32_MAX


class Format(Enum):
    """
    The Enum of format.
    """
    DEFAULT = -1
    NCHW = 0
    NHWC = 1
    NHWC4 = 2
    HWKC = 3
    HWCK = 4
    KCHW = 5
    CKHW = 6
    KHWC = 7
    CHWK = 8
    HW = 9
    HW4 = 10
    NC = 11
    NC4 = 12
    NC4HW4 = 13
    NCDHW = 15
    NWC = 16
    NCW = 17
    NDHWC = 18
    NC8HW8 = 19


data_type_py_cxx_map = {
    DataType.UNKNOWN: _c_lite_wrapper.DataType.kTypeUnknown,
    DataType.BOOL: _c_lite_wrapper.DataType.kNumberTypeBool,
    DataType.INT8: _c_lite_wrapper.DataType.kNumberTypeInt8,
    DataType.INT16: _c_lite_wrapper.DataType.kNumberTypeInt16,
    DataType.INT32: _c_lite_wrapper.DataType.kNumberTypeInt32,
    DataType.INT64: _c_lite_wrapper.DataType.kNumberTypeInt64,
    DataType.UINT8: _c_lite_wrapper.DataType.kNumberTypeUInt8,
    DataType.UINT16: _c_lite_wrapper.DataType.kNumberTypeUInt16,
    DataType.UINT32: _c_lite_wrapper.DataType.kNumberTypeUInt32,
    DataType.UINT64: _c_lite_wrapper.DataType.kNumberTypeUInt64,
    DataType.FLOAT16: _c_lite_wrapper.DataType.kNumberTypeFloat16,
    DataType.FLOAT32: _c_lite_wrapper.DataType.kNumberTypeFloat32,
    DataType.FLOAT64: _c_lite_wrapper.DataType.kNumberTypeFloat64,
    DataType.INVALID: _c_lite_wrapper.DataType.kInvalidType,
}

data_type_cxx_py_map = {
    _c_lite_wrapper.DataType.kTypeUnknown: DataType.UNKNOWN,
    _c_lite_wrapper.DataType.kNumberTypeBool: DataType.BOOL,
    _c_lite_wrapper.DataType.kNumberTypeInt8: DataType.INT8,
    _c_lite_wrapper.DataType.kNumberTypeInt16: DataType.INT16,
    _c_lite_wrapper.DataType.kNumberTypeInt32: DataType.INT32,
    _c_lite_wrapper.DataType.kNumberTypeInt64: DataType.INT64,
    _c_lite_wrapper.DataType.kNumberTypeUInt8: DataType.UINT8,
    _c_lite_wrapper.DataType.kNumberTypeUInt16: DataType.UINT16,
    _c_lite_wrapper.DataType.kNumberTypeUInt32: DataType.UINT32,
    _c_lite_wrapper.DataType.kNumberTypeUInt64: DataType.UINT64,
    _c_lite_wrapper.DataType.kNumberTypeFloat16: DataType.FLOAT16,
    _c_lite_wrapper.DataType.kNumberTypeFloat32: DataType.FLOAT32,
    _c_lite_wrapper.DataType.kNumberTypeFloat64: DataType.FLOAT64,
    _c_lite_wrapper.DataType.kInvalidType: DataType.INVALID,
}

format_py_cxx_map = {
    Format.DEFAULT: _c_lite_wrapper.Format.DEFAULT_FORMAT,
    Format.NCHW: _c_lite_wrapper.Format.NCHW,
    Format.NHWC: _c_lite_wrapper.Format.NHWC,
    Format.NHWC4: _c_lite_wrapper.Format.NHWC4,
    Format.HWKC: _c_lite_wrapper.Format.HWKC,
    Format.HWCK: _c_lite_wrapper.Format.HWCK,
    Format.KCHW: _c_lite_wrapper.Format.KCHW,
    Format.CKHW: _c_lite_wrapper.Format.CKHW,
    Format.KHWC: _c_lite_wrapper.Format.KHWC,
    Format.CHWK: _c_lite_wrapper.Format.CHWK,
    Format.HW: _c_lite_wrapper.Format.HW,
    Format.HW4: _c_lite_wrapper.Format.HW4,
    Format.NC: _c_lite_wrapper.Format.NC,
    Format.NC4: _c_lite_wrapper.Format.NC4,
    Format.NC4HW4: _c_lite_wrapper.Format.NC4HW4,
    Format.NCDHW: _c_lite_wrapper.Format.NCDHW,
    Format.NWC: _c_lite_wrapper.Format.NWC,
    Format.NCW: _c_lite_wrapper.Format.NCW,
    Format.NDHWC: _c_lite_wrapper.Format.NDHWC,
    Format.NC8HW8: _c_lite_wrapper.Format.NC8HW8,
}

format_cxx_py_map = {
    _c_lite_wrapper.Format.DEFAULT_FORMAT: Format.DEFAULT,
    _c_lite_wrapper.Format.NCHW: Format.NCHW,
    _c_lite_wrapper.Format.NHWC: Format.NHWC,
    _c_lite_wrapper.Format.NHWC4: Format.NHWC4,
    _c_lite_wrapper.Format.HWKC: Format.HWKC,
    _c_lite_wrapper.Format.HWCK: Format.HWCK,
    _c_lite_wrapper.Format.KCHW: Format.KCHW,
    _c_lite_wrapper.Format.CKHW: Format.CKHW,
    _c_lite_wrapper.Format.KHWC: Format.KHWC,
    _c_lite_wrapper.Format.CHWK: Format.CHWK,
    _c_lite_wrapper.Format.HW: Format.HW,
    _c_lite_wrapper.Format.HW4: Format.HW4,
    _c_lite_wrapper.Format.NC: Format.NC,
    _c_lite_wrapper.Format.NC4: Format.NC4,
    _c_lite_wrapper.Format.NC4HW4: Format.NC4HW4,
    _c_lite_wrapper.Format.NCDHW: Format.NCDHW,
    _c_lite_wrapper.Format.NWC: Format.NWC,
    _c_lite_wrapper.Format.NCW: Format.NCW,
    _c_lite_wrapper.Format.NDHWC: Format.NDHWC,
    _c_lite_wrapper.Format.NC8HW8: Format.NC8HW8,
}


class Tensor:
    """
    The Tensor class defines a tensor in MindSporeLite.

    Args:
        tensor(Tensor, optional): The data to be stored in a new tensor. It can be another Tensor. Default: None.

    Raises:
        TypeError: `tensor` is neither a Tensor nor None.

    Examples:
        >>> import mindspore_lite as mslite
        >>> tensor = mslite.Tensor()
        >>> tensor.set_data_type(mslite.DataType.FLOAT32)
        >>> print(tensor)
        tensor_name: ,
        data_type: DataType.FLOAT32,
        shape: [],
        format: Format.NCHW,
        element_num: 1,
        data_size: 4.
    """

    def __init__(self, tensor=None):
        self._numpy_obj = None
        if tensor is not None:
            if not isinstance(tensor, _c_lite_wrapper.TensorBind):
                raise TypeError(f"tensor must be TensorBind, but got {type(tensor)}.")
            self._tensor = tensor
        else:
            self._tensor = _c_lite_wrapper.create_tensor()

    def set_tensor_name(self, tensor_name):
        """
        Set the name of the tensor.

        Args:
            tensor_name (str): The name of the tensor.

        Raises:
            TypeError: `tensor_name` is not a str.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_tensor_name("tensor0")
        """
        if not isinstance(tensor_name, str):
            raise TypeError(f"tensor_name must be str, but got {type(tensor_name)}.")
        self._tensor.set_tensor_name(tensor_name)

    def get_tensor_name(self):
        """
        Get the name of the tensor.

        Returns:
            str, the name of the tensor.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_tensor_name("tensor0")
            >>> tensor_name = tensor.get_tensor_name()
            >>> print(tensor_name)
            tensor0
        """
        return self._tensor.get_tensor_name()

    def set_data_type(self, data_type):
        """
        Set data type for the Tensor.

        Args:
            data_type (DataType): The data type of the Tensor.

        Raises:
            TypeError: `data_type` is not a DataType.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_data_type(mslite.DataType.FLOAT32)
        """
        if not isinstance(data_type, DataType):
            raise TypeError(f"data_type must be DataType, but got {type(data_type)}.")
        self._tensor.set_data_type(data_type_py_cxx_map.get(data_type))

    def get_data_type(self):
        """
        Get the data type of the tensor.

        Returns:
            DataType, the data type of the tensor.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_data_type(mslite.DataType.FLOAT32)
            >>> data_type = tensor.get_data_type()
            >>> print(data_type)
            DataType.FLOAT32
        """
        return data_type_cxx_py_map.get(self._tensor.get_data_type())

    def set_shape(self, shape):
        """
        Set shape for the tensor.

        Args:
            shape (list[int]): The shape of the tensor.

        Raises:
            TypeError: `shape` is not a list.
            TypeError: `shape` is a list, but the elements is not int.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_shape([1, 112, 112, 3])
        """
        if not isinstance(shape, list):
            raise TypeError(f"shape must be list, but got {type(shape)}.")
        for i, element in enumerate(shape):
            if not isinstance(element, int):
                raise TypeError(f"shape element must be int, but got {type(element)} at index {i}.")
        self._tensor.set_shape(shape)

    def get_shape(self):
        """
        Get the shape of the tensor.

        Returns:
            list[int], the shape of the tensor.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_shape([1, 112, 112, 3])
            >>> shape = tensor.get_shape()
            >>> print(shape)
            [1, 112, 112, 3]
        """
        return self._tensor.get_shape()

    def set_format(self, tensor_format):
        """
        Set format of the tensor.

        Args:
            tensor_format (Format): The format of the tensor.

        Raises:
            TypeError: `tensor_format` is not a Format.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_format(mslite.Format.NHWC)
        """
        if not isinstance(tensor_format, Format):
            raise TypeError(f"tensor_format must be Format, but got {type(tensor_format)}.")
        self._tensor.set_format(format_py_cxx_map.get(tensor_format))

    def get_format(self):
        """
        Get the format of the tensor.

        Returns:
            Format, the format of the tensor.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_format(mslite.Format.NHWC)
            >>> tensor_format = tensor.get_format()
            >>> print(tensor_format)
            Format.NHWC
        """
        return format_cxx_py_map.get(self._tensor.get_format())

    def get_element_num(self):
        """
        Get the element num of the tensor.

        Returns:
            int, the element num of the tensor data.

        Examples:
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> num = tensor.get_element_num()
            >>> print(num)
            1
        """
        return self._tensor.get_element_num()

    def get_data_size(self):
        """
        Get the data size of the tensor, i.e.,
        data_size = element_num * data_type.

        Returns:
            int, the data size of the tensor data.

        Examples:
            >>> # data_size is related to data_type
            >>> import mindspore_lite as mslite
            >>> tensor = mslite.Tensor()
            >>> tensor.set_data_type(mslite.DataType.FLOAT32)
            >>> size = tensor.get_data_size()
            >>> print(size)
            4
        """
        return self._tensor.get_data_size()

    def set_data_from_numpy(self, numpy_obj):
        """
        Set the data for the tensor from the numpy object.

        Args:
            numpy_obj(numpy.ndarray): the numpy object.

        Raises:
            TypeError: `numpy_obj` is not a numpy.ndarray.
            RuntimeError: The data type of `numpy_obj` is not equivalent to the data type of the tensor.
            RuntimeError: The data size of `numpy_obj` is not equal to the data size of the tensor.

        Examples:
            >>> # in_data download link: https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin
            >>> # 1. set tensor data which is from file
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> tensor = mslite.Tensor()
            >>> tensor.set_shape([1, 224, 224, 3])
            >>> tensor.set_data_type(mslite.DataType.FLOAT32)
            >>> in_data = np.fromfile("input.bin", dtype=np.float32)
            >>> tensor.set_data_from_numpy(in_data)
            >>> print(tensor)
            tensor_name: ,
            data_type: DataType.FLOAT32,
            shape: [1, 224, 224, 3],
            format: Format.NCHW,
            element_num: 150528,
            data_size: 602112.
            >>> # 2. set tensor data which is numpy arange
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> tensor = mslite.Tensor()
            >>> tensor.set_shape([1, 2, 2, 3])
            >>> tensor.set_data_type(mslite.DataType.FLOAT32)
            >>> in_data = np.arange(1 * 2 * 2 * 3, dtype=np.float32)
            >>> tensor.set_data_from_numpy(in_data)
            >>> print(tensor)
            tensor_name: ,
            data_type: DataType.FLOAT32,
            shape: [1, 2, 2, 3],
            format: Format.NCHW,
            element_num: 12,
            data_size: 48.
        """
        if not isinstance(numpy_obj, numpy.ndarray):
            raise TypeError(f"numpy_obj must be numpy.ndarray, but got {type(numpy_obj)}.")
        data_type_map = {
            numpy.bool_: DataType.BOOL,
            numpy.int8: DataType.INT8,
            numpy.int16: DataType.INT16,
            numpy.int32: DataType.INT32,
            numpy.int64: DataType.INT64,
            numpy.uint8: DataType.UINT8,
            numpy.uint16: DataType.UINT16,
            numpy.uint32: DataType.UINT32,
            numpy.uint64: DataType.UINT64,
            numpy.float16: DataType.FLOAT16,
            numpy.float32: DataType.FLOAT32,
            numpy.float64: DataType.FLOAT64,
        }
        if data_type_map.get(numpy_obj.dtype.type) != self.get_data_type():
            raise RuntimeError(
                f"data type not equal! Numpy type: {numpy_obj.dtype.type}, Tensor type: {self.get_data_type()}")
        if numpy_obj.nbytes != self.get_data_size():
            raise RuntimeError(
                f"data size not equal! Numpy size: {numpy_obj.nbytes}, Tensor size: {self.get_data_size()}")
        self._numpy_obj = numpy_obj.flatten()  # keep reference count of numpy objects
        self._tensor.set_data_from_numpy(self._numpy_obj)

    def get_data_to_numpy(self):
        """
        Get the data from the tensor to the numpy object.

        Returns:
            numpy.ndarray, the numpy object from tensor data.

        Examples:
            >>> import mindspore_lite as mslite
            >>> import numpy as np
            >>> tensor = mslite.Tensor()
            >>> tensor.set_shape([1, 2, 2, 3])
            >>> tensor.set_data_type(mslite.DataType.FLOAT32)
            >>> in_data = np.arange(1 * 2 * 2 * 3, dtype=np.float32)
            >>> tensor.set_data_from_numpy(in_data)
            >>> data = tensor.get_data_to_numpy()
            >>> print(data)
            [[[[ 0.  1.  2.]
               [ 3.  4.  5.]]
              [[ 6.  7.  8.]
               [ 9. 10. 11.]]]]
        """
        return self._tensor.get_data_to_numpy()

    def __str__(self):
        res = f"tensor_name: {self.get_tensor_name()},\n" \
              f"data_type: {self.get_data_type()},\n" \
              f"shape: {self.get_shape()},\n" \
              f"format: {self.get_format()},\n" \
              f"element_num: {self.get_element_num()},\n" \
              f"data_size: {self.get_data_size()}."
        return res
