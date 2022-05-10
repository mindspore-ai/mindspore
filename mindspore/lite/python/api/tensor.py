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


class Tensor:
    """
    The MSTensor class defines a tensor in MindSpore.

    Args:
        None

    Raises:
        TypeError: type of input parameters are invalid.

    Examples:
        >>> import mindspore_lite as mslite
        >>> tensor = mslite.tensor.Tensor()
    """

    def __init__(self, tensor=None):
        if tensor is not None:
            if not isinstance(tensor, _c_lite_wrapper.TensorBind):
                raise TypeError(f"tensor must be TensorBind, but got {type(tensor)}.")
            self._tensor = tensor
        else:
            self._tensor = _c_lite_wrapper.TensorBind()

    def set_tensor_name(self, tensor_name):
        """
        Set the name of the tensor.

        Args:
            tensor_name (str): The name of the tensor.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> tensor = mslite.tensor.Tensor()
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
            >>> name = tensor.get_tensor_name()
        """
        return self._tensor.get_tensor_name()

    def set_data_type(self, data_type):
        """
        Set data type for the Tensor.

        Args:
            data_type (DataType): The data type of the Tensor.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> tensor = mslite.tensor.Tensor()
            >>> tensor.set_data_type(mslite.tensor.DataType.FLOAT32)
        """
        if not isinstance(data_type, DataType):
            raise TypeError(f"data_type must be DataType, but got {type(data_type)}.")
        data_type_map = {
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
        self._tensor.set_data_type(data_type_map.get(data_type))

    def get_data_type(self):
        """
        Get the data type of the tensor.

        Returns:
            DataType, the data type of the tensor.

        Examples:
            >>> data_type = tensor.get_data_type()
        """
        data_type_map = {
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
        return data_type_map.get(self._tensor.get_data_type())

    def set_shape(self, shape):
        """
        Set shape for the tensor.

        Args:
            shape (list[int]): The shape of the tensor.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> tensor = mslite.tensor.Tensor()
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
            >>> shape = tensor.get_shape()
        """
        return self._tensor.get_shape()

    def set_format(self, tensor_format):
        """
        Set format of the tensor.

        Args:
            tensor_format (Format): The format of the tensor.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> tensor = mslite.tensor.Tensor()
            >>> tensor.set_format(mslite.tensor.Format.NHWC)
        """
        if not isinstance(tensor_format, Format):
            raise TypeError(f"tensor_format must be Format, but got {type(tensor_format)}.")
        format_map = {
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
        self._tensor.set_format(format_map.get(tensor_format))

    def get_format(self):
        """
        Get the format of the tensor.

        Returns:
            Format, the format of the tensor.

        Examples:
            >>> tensor_format = tensor.get_format()
        """
        format_map = {
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
        return format_map.get(self._tensor.get_format())

    def get_element_num(self):
        """
        Get the element num of the tensor.

        Returns:
            int, the element num of the tensor data.

        Examples:
            >>> num = tensor.get_element_num()
        """
        return self._tensor.get_element_num()

    def get_data_size(self):
        """
        Get the data size of the tensor.

        Returns:
            int, the data size of the tensor data.

        Examples:
            >>> size = tensor.get_data_size()
        """
        return self._tensor.get_data_size()

    def set_data_from_numpy(self, numpy_obj):
        """
        Set the data for the tensor from the numpy object.

        Args:
            numpy_obj(numpy.ndarray): The name of the tensor.

        Raises:
            TypeError: type of input parameters are invalid.

        Examples:
            >>> in_data = numpy.fromfile("model.ms.bin", dtype=np.float32)
            >>> tensor.set_data_from_numpy(in_data)
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
        numpy_obj.flatten()
        self._tensor.set_data_from_numpy(numpy_obj)
        self._numpy_obj = numpy_obj  # keep reference count of numpy objects

    def get_data_to_numpy(self):
        """
        Get the data from the tensor to the numpy object.

        Returns:
            numpy.ndarray, the numpy object from tensor data.

        Examples:
            >>> data = tensor.get_data_to_numpy()
        """
        return self._tensor.get_data_to_numpy()

    def __str__(self):
        res = f"tensor_name: {self.get_tensor_name()}, " \
              f"data_type: {self.get_data_type()}, " \
              f"shape: {self.get_shape()}, " \
              f"format: {self.get_format()}, " \
              f"element_num, {self.get_element_num()}, " \
              f"data_size, {self.get_data_size()}."
        return res
