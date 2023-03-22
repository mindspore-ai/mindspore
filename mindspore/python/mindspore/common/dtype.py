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

import enum
from inspect import isfunction
import numpy as np
from mindspore._c_expression import typing
from mindspore._c_expression.typing import Type

__dtype__ = [
    "int8", "byte",
    "int16", "short",
    "int32", "intc",
    "int64", "intp",
    "uint8", "ubyte",
    "uint16", "ushort",
    "uint32", "uintc",
    "uint64", "uintp",
    "float16", "half",
    "float32", "single",
    "float64", "double",
    "bool_", "float_",
    "list_", "tuple_",
    "int_", "uint",
    "number", "tensor",
    "string", "type_none",
    "tensor_type", "_null",
    "Type", "Int",
    "complex64", "complex128"
]

__method__ = [
    "dtype_to_nptype", "dtype_to_pytype",
    "pytype_to_dtype", "get_py_obj_dtype"
]

__all__ = ["Type", "QuantDtype"]
__all__.extend(__dtype__)
__all__.extend(__method__)

# type definition
bool_ = typing.Bool()

int8 = typing.Int(8)
byte = int8
int16 = typing.Int(16)
short = int16
int32 = typing.Int(32)
intc = int32
int64 = typing.Int(64)
intp = int64

uint8 = typing.UInt(8)
ubyte = uint8
uint16 = typing.UInt(16)
ushort = uint16
uint32 = typing.UInt(32)
uintc = uint32
uint64 = typing.UInt(64)
uintp = uint64

float16 = typing.Float(16)
half = float16
float32 = typing.Float(32)
single = float32
float64 = typing.Float(64)
double = float64
complex64 = typing.Complex(64)
complex128 = typing.Complex(128)

number = typing.Number()
int_ = typing.Int()
uint = typing.UInt()
float_ = typing.Float()
string = typing.String()
list_ = typing.List()
tuple_ = typing.Tuple()
type_none = typing.TypeNone()
_null = typing.TypeNull()

tensor = typing.TensorType()
index_slices = typing.RowTensorType()
coo_tensor = typing.COOTensorType()
csr_tensor = typing.CSRTensorType()
undetermined = typing.UndeterminedType()

function = typing.Function()
symbolic_key = typing.SymbolicKeyType()
env_type = typing.EnvType()
type_type = typing.TypeType()
type_refkey = typing.RefKeyType()

Int = typing.Int
Float = typing.Float
Bool = typing.Bool
String = typing.String
List = typing.List
Tuple = typing.Tuple
Dict = typing.Dict
Slice = typing.Slice
function_type = typing.Function
Ellipsis_ = typing.TypeEllipsis
MsClassType = typing.TypeMsClassType
none_type = typing.TypeNone
env_type_type = typing.EnvType
tensor_type = typing.TensorType
csr_tensor_type = typing.CSRTensorType
anything_type = typing.TypeAnything
ref_type = typing.RefType
_null_type = typing.TypeNull

number_type = (int8,
               int16,
               int32,
               int64,
               uint8,
               uint16,
               uint32,
               uint64,
               float16,
               float32,
               float64,
               complex64,
               complex128,)

int_type = (int8, int16, int32, int64,)
uint_type = (uint8, uint16, uint32, uint64,)
float_type = (float16, float32, float64,)
signed_type = (int8, byte, int16, short, int32, intc, int64,
               intp, float16, half, float32, single, float64,
               double, complex64, complex128)
complex_type = (complex64, complex128,)
all_types = (bool_, int8, uint8, int16, int32, int64, float16, float32, float64, complex64, complex128)
implicit_conversion_seq = {t: idx for idx, t in enumerate(all_types)}

_simple_types = {
    list: list_,
    tuple: tuple_,
    type(None): type_none,
    bool: bool_,
    int: int64,
    float: float64,
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

    Args:
        obj (type): A python type object.

    Returns:
        Type of MindSpore type.

    Raises:
        NotImplementedError: If the python type cannot be converted to MindSpore type.
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

    Args:
        obj (type): An object of python type, or a variable of python type.

    Returns:
        Type of MindSpore type.
    """
    # Tensor
    if hasattr(obj, 'shape') and hasattr(obj, 'dtype') and isinstance(obj.dtype, typing.Type):
        return tensor_type(obj.dtype)
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
    """
    Convert MindSpore dtype to numpy data type.

    Args:
        type_ (:class:`mindspore.dtype`): MindSpore's dtype.

    Returns:
        The data type of numpy.
    """

    return {
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
    }[type_]


def dtype_to_pytype(type_):
    """
    Convert MindSpore dtype to python data type.

    Args:
        type_ (:class:`mindspore.dtype`): MindSpore's dtype.

    Returns:
        Type of python.
    """

    return {
        bool_: bool,
        int_: int,
        int8: int,
        int16: int,
        int32: int,
        int64: int,
        uint8: int,
        uint16: int,
        uint32: int,
        uint64: int,
        float_: float,
        float16: float,
        float32: float,
        float64: float,
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

    def value(self) -> int:
        """
        Return value of `QuantDtype`.

        Returns:
            An int as value of `QuantDtype`.
        """
        return self._value_
