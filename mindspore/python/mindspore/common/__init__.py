# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
from mindspore.graph.api import ms_memory_recycle, jit, jit_class, _no_grad, saved_tensors_hooks, \
    register_saved_tensors_hooks, flops_collection, set_recursion_limit
from mindspore.common.dtype import Type, int8, byte, int16, short, int, int32, intc, long, int64, intp, \
    uint8, ubyte, uint16, ushort, uint32, uintc, uint64, uintp, float16, half, \
    float, float32, single, float64, bfloat16, double, bool, bool_, float_, list_, tuple_, int_, \
    uint, number, tensor_type, string, type_none, TensorType, Int, \
    cfloat, complex64, cdouble, complex128, dtype_to_nptype, _null, _NullType, \
    dtype_to_pytype, pytype_to_dtype, get_py_obj_dtype, QuantDtype, qint4x2, \
    float8_e4m3fn, float8_e5m2, hifloat8
from mindspore.common.dump import set_dump
from mindspore.common.file_system import set_mindio_server_info, mindio_preload
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.seed import set_seed, get_seed
from mindspore.common.tensor import Tensor, tensor
from mindspore.common.storage import UntypedStorage
from mindspore.common.sparse_tensor import RowTensor, RowTensorInner, SparseTensor, COOTensor, CSRTensor
from mindspore.graph.mutable import mutable
from mindspore.graph.jit_config import JitConfig
from mindspore.graph.lazy_inline import lazy_inline
from mindspore.graph.no_inline import no_inline
from mindspore.graph._utils.mindir_util import load_mindir, save_mindir
from mindspore.graph.symbol import Symbol
from mindspore.common.recompute import recompute
from mindspore.common import generator
from mindspore.common.generator import (
    Generator, default_generator, seed, manual_seed, initial_seed, get_rng_state, set_rng_state)
from mindspore.ops.function.array_func import is_tensor, from_numpy
from mindspore.common._grad_function import _Function
from mindspore.graph.dynamic_shape.enable_dynamic import enable_dynamic

# Import the following modules to maintain compatibility
import mindspore.graph._utils.mindir_util as mindir_util
import mindspore.graph._utils.monad as _monad
import mindspore.graph._utils.pijit_context as _pijit_context
import mindspore.graph.api as api
import mindspore.graph.dynamic_shape as dynamic_shape
import mindspore.graph.jit_config as jit_config
import mindspore.graph.jit_context as jit_context
import mindspore.graph.jit_trace as jit_trace
import mindspore.graph.symbol as symbol

try:
    import triton
    if isinstance(getattr(triton.runtime.jit, "type_canonicalisation_dict", None), dict):
        ms_type_canonicalisation_dict = {
            "Bool": "i1",
            "Float16": "fp16",
            "BFloat16": "bf16",
            "Float32": "fp32",
            "Float64": "fp64",
            "Int8": "i8",
            "Int16": "i16",
            "Int32": "i32",
            "Int64": "i64",
            "UInt8": "u8",
            "UInt16": "u16",
            "UInt32": "u32",
            "UInt64": "u64",
        }
        triton.runtime.jit.type_canonicalisation_dict.update(ms_type_canonicalisation_dict)

except ImportError:
    pass

# symbols from dtype
# bool, int, float are not defined in __all__ to avoid conflict with built-in types.
__all__ = [
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
    "float_", "list_", "tuple_",
    "int_", "uint",
    "number", "tensor_type",
    "string", "type_none",
    "_null",
    "TensorType", "QuantDtype",
    "Type", "Int", "_NullType",
    "complex64", "cfloat",
    "complex128", "cdouble",
    "bfloat16", "qint4x2",
    "float8_e4m3fn", "float8_e5m2", "hifloat8",
    # __method__ from dtype
    "dtype_to_nptype", "dtype_to_pytype",
    "pytype_to_dtype", "get_py_obj_dtype"
]

__all__.extend([
    "tensor", "Tensor", "RowTensor", "SparseTensor", "COOTensor", "CSRTensor",  # tensor
    'jit', 'jit_class', '_no_grad', 'saved_tensors_hooks', 'register_saved_tensors_hooks',  # api
    "Parameter", "ParameterTuple",  # parameter
    "UntypedStorage",
    "dtype",
    "set_seed", "get_seed", "manual_seed", # random seed
    "set_dump",
    "ms_memory_recycle",
    "set_recursion_limit",
    "mutable", "JitConfig",
    "enable_dynamic",
    "flops_collection",
    "lazy_inline", "load_mindir", "save_mindir",
    "no_inline",
    "Symbol",
    "recompute",
    "is_tensor", "from_numpy", "_Function",
    "set_mindio_server_info", "mindio_preload"
])
__all__.extend(generator.__all__)
