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
"""Top-level reference to dtype of common module."""
from __future__ import absolute_import
from mindspore.common import dtype
from mindspore.common.api import ms_function, ms_memory_recycle, ms_class, jit, jit_class
from mindspore.common.dtype import Type, int8, byte, int16, short, int32, intc, int64, intp, \
    uint8, ubyte, uint16, ushort, uint32, uintc, uint64, uintp, float16, half, \
    float32, single, float64, double, bool_, float_, list_, tuple_, int_, \
    uint, number, tensor, string, type_none, tensor_type, Int, \
    complex64, complex128, dtype_to_nptype, _null, _null_type, \
    dtype_to_pytype, pytype_to_dtype, get_py_obj_dtype, QuantDtype
from mindspore.common.dump import set_dump
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.seed import set_seed, get_seed
from mindspore.common.tensor import Tensor
from mindspore.common.sparse_tensor import RowTensor, RowTensorInner, SparseTensor, COOTensor, CSRTensor
from mindspore.common.mutable import mutable
from mindspore.common.jit_config import JitConfig

# symbols from dtype
__all__ = [
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
    "_null",
    "tensor_type", "QuantDtype",
    "Type", "Int", "_null_type",
    "complex64", "complex128",
    # __method__ from dtype
    "dtype_to_nptype", "dtype_to_pytype",
    "pytype_to_dtype", "get_py_obj_dtype"
]

__all__.extend([
    "Tensor", "RowTensor", "SparseTensor", "COOTensor", "CSRTensor", # tensor
    "ms_function", "ms_class", 'jit', 'jit_class',  # api
    "Parameter", "ParameterTuple",  # parameter
    "dtype",
    "set_seed", "get_seed",  # random seed
    "set_dump",
    "ms_memory_recycle",
    "mutable", "JitConfig",
])
