# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""The module of parser python object, called by c++."""

import collections
import types
import math
import os
import numpy
from mindspore.nn import GraphCell, Cell
from mindspore.ops.primitive import Primitive, constexpr, _primexpr
from mindspore.ops.composite.base import GradOperation, _Grad
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.api import jit
from mindspore.common.tensor import Tensor
from mindspore.common._register_for_tensor import Registry, tensor_operator_registry
from mindspore._c_expression import MetaFuncGraph_, function_id, Primitive_, PrimitiveFunction_
from mindspore._c_expression import Tensor as Tensor_
from mindspore._extends.parse.resources import convert_object_map
from mindspore import _checkparam as validator


def _get_after_grad_code():
    """Get the code object of 'after_grad'"""
    name = "after_grad"
    codes = []
    for cnst in GradOperation.__call__.__code__.co_consts:
        if isinstance(cnst, types.CodeType) and cnst.co_name == name:
            codes.append(cnst)
    for cnst in _Grad.__call__.__code__.co_consts:
        if isinstance(cnst, types.CodeType) and cnst.co_name == name:
            codes.append(cnst)
    if not codes:
        raise RuntimeError("check GradOperation, can't find the code of 'after_grad'")
    return codes


def _get_psjit_code():
    """Get the code object of 'staging_specialize'"""
    @jit
    def inner():
        pass
    return inner.__code__


def _get_constexpr_code():
    """Get the code object of '@constexpr'"""
    @constexpr
    def inner():
        pass
    code = inner.__call__.__code__
    # check it before c++ use it
    if not isinstance(inner, Primitive) or code is Primitive.__call__.__code__:
        raise RuntimeError("@constexpr not isinstance(inner, Primitive) or code is Primitive.__call__.__code__")
    return code


def _get_primexpr_code():
    """Get the code object of '@_primexpr'"""
    @_primexpr
    def inner():
        pass
    code = inner.__call__.__code__
    # check it before c++ use it
    if not isinstance(inner, Primitive) or code is Primitive.__call__.__code__:
        raise RuntimeError("@_primexpr not isinstance(inner, Primitive) or code is Primitive.__call__.__code__")
    return code


def _pijit_constexpr():
    """Placeholder for uniqure id"""

def _get_pijit_constexpr_code():
    codes = []
    for cnst in validator.check_transpose_axis.__code__.co_consts:
        if isinstance(cnst, types.CodeType) and cnst.co_name == "_check_dim":
            codes.append(cnst)
    return codes

def _get_ms_api():
    """Get ms api"""
    target_types = Cell, types.FunctionType, Primitive_, PrimitiveFunction_
    results = []
    from mindspore.ops import operations as P
    from mindspore.ops import functional as F
    from mindspore.ops import composite as C
    mods = P, F, C
    for mod in mods:
        for i in mod.__all__:
            f = getattr(mod, i)
            if isinstance(f, target_types):
                results.append(f)
    for f in tensor_operator_registry.values():
        if isinstance(f, target_types):
            results.append(f)
    return results


psjit_code = _get_psjit_code()
constexpr_code = _get_constexpr_code()
primexpr_code = _get_primexpr_code()

primitive_key = id(Primitive.__call__)
constexpr_key = id(constexpr_code)
primexpr_key = id(primexpr_code)
meta_func_graph_key = id(MetaFuncGraph_)
pijit_forbidden_key = id(NotImplemented)
pijit_constexpr_key = id(_pijit_constexpr)


# check WrapperDescriptor: function_id(tuple.__getitem__) == function_id(tuple().__getitem__)
# check MethodDescriptor: function_id(list.__getitem__) == function_id(list().__getitem__)
# check instancemethod: function_id(Tensor_.from_numpy) == function_id(Tensor_(1).from_numpy)
# check cfunction filter: function_id(Tensor_.from_numpy) != function_id(Tensor_._is_test_stub)
# check function id: function_id(Tensor.astype) == function_id(Tensor(1).astype) == id(Tensor.astype)
# check user defined object id: function_id(Primitive) == function_id(Primitive) == id(Primitive)


FUNC_KEY_EMPTY = 0  # ""
FUNC_KEY_PIJIT_CONSTEXPR = 1  # "pijit.constexpr"
FUNC_KEY_PIJIT_FORBIDDEN = 2  # "pijit.forbidden"
FUNC_KEY_BUILTIN_FUNC = 3  # "builtin.func"
FUNC_KEY_LIST_APPEND = 4  # "list.append"
FUNC_KEY_DICT_POP = 5  # "dict.pop"
FUNC_KEY_PRIMITIVE = 6  # "mindspore._c_expression.Primitive_"
FUNC_KEY_META_FUNCG_RAPH = 7  # "mindspore._c_expression.MetaFuncGraph_"
FUNC_KEY_PSJIT_CODE = 8  # "mindspore.common.api.jit.<locals>.staging_specialize"
FUNC_KEY_CONSTEXPR = 9  # "mindspore.ops.primitive.constexpr"
FUNC_KEY_PRIMEXPR = 10  # "mindspore.ops.primitive._primexpr"
FUNC_KEY_GET_CACHE_PRIM = 11  # "mindspore.ops._primitive_cache._get_cache_prim"
FUNC_KEY_REGISTRY_GET = 12  # "mindspore.common._register_for_tensor.Registry.get"
FUNC_KEY_TENSOR_ASTYPE = 13  # "mindspore.common.tensor.Tensor.astype"
FUNC_KEY_GRAD_OPERATIONS_CODE = 14 # "mindspore.ops.composite.base._Grad.__call__.<locals>.after_grad"
FUNC_KEY_PSJIT_CONVERTMAP = 15 # "mindspore._extends.parse.resources.convert_object_map"
FUNC_KEY_GRAPH_CELL = 16  # "mindspore.nn.cell.GraphCell"
FUNC_KEY_MS_API = 17  # mindspore common api
FUNC_KEY_MAPPING_GET = 18 # collections.abc.Mapping.get

# Initialized only once. This map will initialize by c++ when start pijit.
# key is customer if fuzzy match. (Primitive, constexpr, primexpr, MetaFuncGraph)
# key is id of code for nest object. (jit.<locals>.staging_specialize, GradOperation.__call__.<locals>.after_grad)
# key is id of object for callalbe object.
# key is cfunction pointer for builtin_function or method. (isinstance, tuple.__getitem__, Tensor_.asnumpy)
_func_map = {
    # special function
    pijit_constexpr_key: FUNC_KEY_PIJIT_CONSTEXPR,
    pijit_forbidden_key: FUNC_KEY_PIJIT_FORBIDDEN,
    primitive_key: FUNC_KEY_PRIMITIVE,
    constexpr_key: FUNC_KEY_CONSTEXPR,
    primexpr_key: FUNC_KEY_PRIMEXPR,
    meta_func_graph_key: FUNC_KEY_META_FUNCG_RAPH,
    id(GraphCell.__call__): FUNC_KEY_GRAPH_CELL,
    id(psjit_code): FUNC_KEY_PSJIT_CODE,
    id(_get_cache_prim): FUNC_KEY_GET_CACHE_PRIM,
    id(Registry.get): FUNC_KEY_REGISTRY_GET,

    # Tensor method
    id(Tensor.astype): FUNC_KEY_TENSOR_ASTYPE,

    # types.BuiltinFunctionType
    function_id(isinstance): FUNC_KEY_BUILTIN_FUNC,
    function_id(issubclass): FUNC_KEY_BUILTIN_FUNC,
    function_id(len): FUNC_KEY_BUILTIN_FUNC,
    function_id(abs): FUNC_KEY_BUILTIN_FUNC,
    function_id(max): FUNC_KEY_BUILTIN_FUNC,
    function_id(all): FUNC_KEY_BUILTIN_FUNC,
    function_id(any): FUNC_KEY_BUILTIN_FUNC,
    function_id(hash): FUNC_KEY_BUILTIN_FUNC,
    function_id(id): FUNC_KEY_BUILTIN_FUNC,
    function_id(ord): FUNC_KEY_BUILTIN_FUNC,
    function_id(callable): FUNC_KEY_BUILTIN_FUNC,
    function_id(getattr): FUNC_KEY_BUILTIN_FUNC,
    function_id(hasattr): FUNC_KEY_BUILTIN_FUNC,

    # types.MethodDescriptorType, types.WrapperDescriptorType
    function_id(tuple.__getitem__): FUNC_KEY_BUILTIN_FUNC,
    function_id(tuple.count): FUNC_KEY_BUILTIN_FUNC,
    function_id(tuple.index): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.__getitem__): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.copy): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.index): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.count): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.__contains__): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.__getitem__): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.get): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.keys): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.values): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.items): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.fromkeys): FUNC_KEY_BUILTIN_FUNC,
    function_id(dict.copy): FUNC_KEY_BUILTIN_FUNC,
    function_id(set.__contains__): FUNC_KEY_BUILTIN_FUNC,
    function_id(set.copy): FUNC_KEY_BUILTIN_FUNC,
    function_id(set.issubset): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.find): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.count): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.index): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.rfind): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.rindex): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.startswith): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.endswith): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isascii): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.islower): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isupper): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.istitle): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isspace): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isdecimal): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isdigit): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isnumeric): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isalpha): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isalnum): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isidentifier): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.isprintable): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.format): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.format_map): FUNC_KEY_BUILTIN_FUNC,
    function_id(str.__format__): FUNC_KEY_BUILTIN_FUNC,
    function_id(list.append): FUNC_KEY_LIST_APPEND,
    function_id(dict.pop): FUNC_KEY_DICT_POP,

    # instancemethod
    function_id(Tensor_._flatten_tensors): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_._is_flattened): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_._get_flattened_tensors): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_._get_fusion_size): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_._is_test_stub): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_.__str__): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_.__repr__): FUNC_KEY_BUILTIN_FUNC,  # pylint: disable=protected-access
    function_id(Tensor_.convert_bytes_to_tensor): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.dim): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.from_numpy): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.getitem_index_info): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.get_bytes): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.is_init): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.is_contiguous): FUNC_KEY_BUILTIN_FUNC,
    function_id(Tensor_.stride): FUNC_KEY_BUILTIN_FUNC,
    # Tensor_.asnumpy need real tensor value

    # other builtin function
    function_id(collections.abc.Mapping.get): FUNC_KEY_MAPPING_GET,
    function_id(math.log): FUNC_KEY_BUILTIN_FUNC,

    function_id(numpy.isinf): FUNC_KEY_BUILTIN_FUNC,
    function_id(numpy.isnan): FUNC_KEY_BUILTIN_FUNC,
    function_id(numpy.abs): FUNC_KEY_BUILTIN_FUNC,
    function_id(numpy.log): FUNC_KEY_BUILTIN_FUNC,

    # const function
    function_id(os.getenv): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(validator.check_number_range): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(validator.check_is_int): FUNC_KEY_PIJIT_CONSTEXPR,
    function_id(validator.check_is_number): FUNC_KEY_PIJIT_CONSTEXPR,
}

for after_grad in _get_after_grad_code():
    _func_map[id(after_grad)] = FUNC_KEY_GRAD_OPERATIONS_CODE

for k, v in convert_object_map.items():
    key = id(k)
    if key not in _func_map and isinstance(v, Primitive):
        if key is print:
            continue
        _func_map[key] = FUNC_KEY_PSJIT_CONVERTMAP

for const_code in _get_pijit_constexpr_code():
    _func_map[id(const_code)] = FUNC_KEY_PIJIT_CONSTEXPR

GUARD_KEY_RELAX_FUNC = 1
_guard_func_map = dict()
