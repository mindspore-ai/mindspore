# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Data type for MindSpore."""
from __future__ import absolute_import

import builtins
import enum
from inspect import isfunction
import numpy as np
from mindspore import log as logger
from mindspore._c_expression import typing
from mindspore._c_expression.typing import Type
from mindspore._c_expression.np_dtypes import np_dtype_valid

if np_dtype_valid(False):
    from mindspore._c_expression.np_dtypes import bfloat16 as np_bfloat16

# bool, int, float are not defined in __all__ to avoid conflict with built-in types.
__dtype__ = [
    "bool_",
    "int8", "byte",
    "int16", "short",
    "int32", "intc",
    "int64", "long", "intp",
    "uint8", "ubyte",
    "uint16", "ushort",
    "uint32", "uintc",
    "uint64", "uintp",
    "float16", "half",
    "float32", "single",
    "float64", "double",
    "complex64", "cfloat",
    "complex128", "cdouble",
    "qint4x2", "bfloat16",
    "float8_e4m3fn", "float8_e5m2", "hifloat8",
    "int_", "uint", "float_",
    "list_", "tuple_", "string",
    "number", "tensor_type",
    "type_none", "_null",
    "TensorType", "Type", "Int",
]

__method__ = [
    "dtype_to_nptype", "dtype_to_pytype",
    "pytype_to_dtype", "get_py_obj_dtype"
]

__all__ = ["Type", "QuantDtype"]
__all__.extend(__dtype__)
__all__.extend(__method__)

# type definition
bool = typing.kBool
bool_ = bool

int8 = typing.kInt8
byte = int8
int16 = typing.kInt16
short = int16
int32 = typing.kInt32
int = int32
intc = int32
int64 = typing.kInt64
long = int64
intp = int64

uint8 = typing.kUInt8
ubyte = uint8
uint16 = typing.kUInt16
ushort = uint16
uint32 = typing.kUInt32
uintc = uint32
uint64 = typing.kUInt64
uintp = uint64

float16 = typing.kFloat16
half = float16
float32 = typing.kFloat32
float = float32
single = float32
float64 = typing.kFloat64
double = float64

qint4x2 = typing.kInt4
float8_e4m3fn = typing.kFloat8E4M3FN
float8_e5m2 = typing.kFloat8E5M2
hifloat8 = typing.kHiFloat8
bfloat16 = typing.kBFloat16

complex64 = typing.kComplex64
cfloat = complex64
complex128 = typing.kComplex128
cdouble = complex128

number = typing.kNumber
int_ = typing.kInt
uint = typing.kUInt
float_ = typing.kFloat
string = typing.kString
list_ = typing.kList
tuple_ = typing.kTuple
type_none = typing.kTypeNone
_null = typing.kTypeNull

tensor_type = typing.kTensorType
index_slices = typing.kRowTensorType
coo_tensor = typing.kCOOTensorType
csr_tensor = typing.kCSRTensorType
undetermined = typing.UndeterminedType()

function = typing.Function()
symbolic_key = typing.SymbolicKeyType()
env_type = typing.kTypeEnv
type_type = typing.kTypeType
type_refkey = typing.kRefKeyType

Int = typing.Int
Float = typing.Float
Bool = typing.Bool
String = typing.String
List = typing.List
Tuple = typing.Tuple
Dict = typing.Dict
Slice = typing.Slice
FunctionType = typing.Function
Ellipsis_ = typing.TypeEllipsis
MsClassType = typing.TypeMsClassType
NoneType = typing.TypeNone
EnvType = typing.EnvType
TensorType = typing.TensorType
CSRTensorType = typing.CSRTensorType
AnythingType = typing.TypeAny
RefType = typing.RefType
_NullType = typing.TypeNull

number_type = (int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16, complex64,
               complex128, qint4x2, float8_e4m3fn, float8_e5m2, hifloat8)

int_type = (int8, int16, int32, int64,)
uint_type = (uint8, uint16, uint32, uint64,)
float_type = (float16, float32, float64, bfloat16, float8_e4m3fn, float8_e5m2, hifloat8)
signed_type = (int8, int16, int32, int64, float16, float32, float64, bfloat16, complex64, complex128, qint4x2,
               float8_e4m3fn, float8_e5m2, hifloat8)
complex_type = (complex64, complex128,)
all_types = (bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, bfloat16,
             complex64, complex128, qint4x2, float8_e4m3fn, float8_e5m2, hifloat8)

_simple_types = {
    list: list_,
    tuple: tuple_,
    type(None): type_none,
    builtins.bool: bool_,
    builtins.int: int64,
    builtins.float: float64,
    complex: complex128,
    str: string,
    np.bool_: bool_,
    np.str_: string,
    np.int8: int8,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.uint64: uint64,
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
}


def pytype_to_dtype(obj):
    """
    Convert python type to MindSpore type.

    Note:
        The interface is deprecated from version 2.7 and will be removed in a future version.

    Args:
        obj (type): A python type object.

    Returns:
        Type of MindSpore type.

    Raises:
        NotImplementedError: If the python type cannot be converted to MindSpore type.

    Examples:
        >>> import mindspore as ms
        >>> out = ms.pytype_to_dtype(bool)
        >>> print(out)
        Bool
    """
    logger.warning("The interface 'mindspore.pytype_to_dtype' is deprecated from version 2.7 "
                   "and will be removed in a future version.")
    return _pytype_to_dtype(obj)


def _pytype_to_dtype(obj):
    """
    Convert python type to MindSpore type.
    """

    if isinstance(obj, np.dtype):
        obj = obj.type
    if isinstance(obj, typing.Type):
        return obj
    if not isinstance(obj, type):
        raise TypeError("For 'pytype_to_dtype', the argument 'obj' must be a python type object,"
                        "such as int, float, str, etc. But got type {}.".format(type(obj)))
    if obj in _simple_types:
        return _simple_types[obj]
    raise NotImplementedError(f"The python type {obj} cannot be converted to MindSpore type.")


def get_py_obj_dtype(obj):
    """
    Get the MindSpore data type, which corresponds to python type or variable.

    Note:
        The interface is deprecated from version 2.7 and will be removed in a future version.

    Args:
        obj (type): An object of python type, or a variable of python type.

    Returns:
        Type of MindSpore type.

    Examples:
        >>> import mindspore as ms
        >>> ms.get_py_obj_dtype(1)
        mindspore.int64
    """
    logger.warning("The interface 'mindspore.get_py_obj_dtype' is deprecated from version 2.7 "
                   "and will be removed in a future version.")
    return _get_py_obj_dtype(obj)


def _get_py_obj_dtype(obj):
    """
    Get the MindSpore data type, which corresponds to python type or variable.
    """
    # Tensor
    if hasattr(obj, 'shape') and hasattr(obj, 'dtype') and isinstance(obj.dtype, typing.Type):
        return TensorType(obj.dtype)
    # Primitive or Cell
    if hasattr(obj, '__primitive_flag__') or hasattr(obj, 'construct'):
        return function
    # python function type
    if isfunction(obj):
        return function
    # mindspore type
    if isinstance(obj, typing.Type):
        return type_type
    # python type
    if isinstance(obj, type):
        return pytype_to_dtype(obj)
    # others
    return pytype_to_dtype(type(obj))


def dtype_to_nptype(type_):
    r"""
    Convert MindSpore dtype to numpy data type.

    Note:
        The interface is deprecated from version 2.7 and will be removed in a future version.

    Args:
        type\_ (:class:`mindspore.dtype`): MindSpore's dtype.

    Returns:
        The data type of numpy.

    Examples:
        >>> import mindspore as ms
        >>> ms.dtype_to_nptype(ms.int8)
        <class 'numpy.int8'>
    """
    logger.warning("The interface 'mindspore.dtype_to_nptype' is deprecated from version 2.7 "
                   "and will be removed in a future version.")
    return _dtype_to_nptype(type_)


def _dtype_to_nptype(type_):
    """
    Convert MindSpore dtype to numpy data type.
    """
    _dtype_nptype_dict = {
        bool_: np.bool_,
        int8: np.int8,
        int16: np.int16,
        int32: np.int32,
        int64: np.int64,
        uint8: np.uint8,
        uint16: np.uint16,
        uint32: np.uint32,
        uint64: np.uint64,
        float16: np.float16,
        float32: np.float32,
        float64: np.float64,
        complex64: np.complex64,
        complex128: np.complex128,
    }
    if type_ == bfloat16:
        if not np_dtype_valid(True):
            raise TypeError(
                "The Numpy bfloat16 data type is not supported now, please ensure that the current "
                "Numpy version is not less than the version when the mindspore is compiled, "
                "and the major versions are same."
            )
        return np_bfloat16
    return _dtype_nptype_dict[type_]


def dtype_to_pytype(type_):
    r"""
    Convert MindSpore dtype to python data type.

    Note:
        The interface is deprecated from version 2.7 and will be removed in a future version.

    Args:
        type\_ (:class:`mindspore.dtype`): MindSpore's dtype.

    Returns:
        Type of python.

    Examples:
        >>> import mindspore as ms
        >>> out = ms.dtype_to_pytype(ms.bool_)
        >>> print(out)
        <class 'bool'>
    """
    logger.warning("The interface 'mindspore.dtype_to_pytype' is deprecated from version 2.7 "
                   "and will be removed in a future version.")
    return _dtype_to_pytype(type_)


def _dtype_to_pytype(type_):
    """
    Convert MindSpore dtype to python data type.
    """
    return {
        bool_: builtins.bool,
        int_: builtins.int,
        int8: builtins.int,
        int16: builtins.int,
        int32: builtins.int,
        int64: builtins.int,
        uint8: builtins.int,
        uint16: builtins.int,
        uint32: builtins.int,
        uint64: builtins.int,
        float_: builtins.float,
        float16: builtins.float,
        float32: builtins.float,
        float64: builtins.float,
        bfloat16: builtins.float,
        list_: list,
        tuple_: tuple,
        string: str,
        complex64: complex,
        complex128: complex,
        type_none: type(None)
    }[type_]


def _issubclass_(type_, dtype):
    if not isinstance(type_, typing.Type):
        return False
    return typing.is_subclass(type_, dtype)


def type_size_in_bytes(dtype):
    """
    Return type size in bytes.

    Args:
        dtype (:class:`mindspore.dtype`): MindSpore dtype.

    Returns:
        Type size in bytes.
    """

    if not isinstance(dtype, typing.Type):
        raise TypeError("The argument `dtype` should be instance of ", typing.Type)
    return typing.type_size_in_bytes(dtype)


@enum.unique
class QuantDtype(enum.Enum):
    """
    An enum for quant datatype, contains `INT1` ~ `INT16`, `UINT1` ~ `UINT16`.

    `QuantDtype` is defined in
    `dtype.py <https://atomgit.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/common/dtype.py>`_ .
    Use command below to import:

    .. code-block::

        from mindspore import QuantDtype

    Tutorial Examples:
        - `Quantization algorithm in Golden Stick
          <https://www.mindspore.cn/golden_stick/docs/en/master/quantization/slb.html
          #applying-the-quantization-algorithm>`_
    """
    INT1 = 0
    INT2 = 1
    INT3 = 2
    INT4 = 3
    INT5 = 4
    INT6 = 5
    INT7 = 6
    INT8 = 7
    INT9 = 8
    INT10 = 9
    INT11 = 10
    INT12 = 11
    INT13 = 12
    INT14 = 13
    INT15 = 14
    INT16 = 15

    UINT1 = 100
    UINT2 = 101
    UINT3 = 102
    UINT4 = 103
    UINT5 = 104
    UINT6 = 105
    UINT7 = 106
    UINT8 = 107
    UINT9 = 108
    UINT10 = 109
    UINT11 = 110
    UINT12 = 111
    UINT13 = 112
    UINT14 = 113
    UINT15 = 114
    UINT16 = 115

    def __str__(self):
        return f"{self.name}"

    def value(self) -> builtins.int:
        """
        Return value of `QuantDtype`. This interface is currently used to serialize or deserialize `QuantDtype`
        primarily.

        Returns:
            An int as value of `QuantDtype`.

        Examples:
            >>> from mindspore import QuantDtype
            >>> print(QuantDtype.INT8.value())
            7
            >>> print(QuantDtype.UINT16.value())
            115
        """
        return self._value_
