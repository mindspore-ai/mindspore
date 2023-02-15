# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Resources for ast tree parse."""
from __future__ import absolute_import

import ast
import math

from mindspore import RowTensor, SparseTensor, COOTensor, CSRTensor
from mindspore.experimental import MapParameter
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.ops import functional as F, composite as C
from mindspore.ops import Primitive
from mindspore.ops.composite import multitype_ops
from mindspore._c_expression import security
from . import standard_method as M
from . import trope as T
from .namespace import CellNamespace

# namespace define
functional_ns = CellNamespace('mindspore.ops.functional')
composite_ns = CellNamespace('mindspore.ops.composite')
trope_ns = CellNamespace('mindspore._extends.parse.trope')

SYMBOL_UNDEFINE = 0xFF      # Undefined var and function

# Some space set aside for readability of code
parse_object_map = {
    # ast grammar
    ast.Add:        (trope_ns, 'add'),
    ast.Sub:        (trope_ns, 'sub'),
    ast.Mult:       (trope_ns, 'mul'),
    ast.Div:        (trope_ns, 'truediv'),
    ast.FloorDiv:   (trope_ns, 'floordiv'),
    ast.Mod:        (trope_ns, 'mod'),
    ast.Pow:        (trope_ns, 'pow'),
    ast.MatMult:    (trope_ns, 'matmul'),
    ast.LShift:     (trope_ns, 'lshift'),
    ast.RShift:     (trope_ns, 'rshift'),
    ast.BitAnd:     (trope_ns, 'and_'),
    ast.BitOr:      (trope_ns, 'or_'),
    ast.BitXor:     (trope_ns, 'xor'),
    ast.UAdd:       (trope_ns, 'pos'),
    ast.USub:       (trope_ns, 'neg'),
    ast.Invert:     (trope_ns, 'invert'),
    ast.Not:        (trope_ns, 'not_'),
    ast.Eq:         (trope_ns, 'eq'),
    ast.NotEq:      (trope_ns, 'ne'),
    ast.Lt:         (trope_ns, 'lt'),
    ast.Gt:         (trope_ns, 'gt'),
    ast.LtE:        (trope_ns, 'le'),
    ast.GtE:        (trope_ns, 'ge'),
    ast.Is:         (trope_ns, 'is_'),
    ast.IsNot:      (trope_ns, 'is_not'),
    ast.In:         (trope_ns, 'contains'),
    ast.NotIn:      (trope_ns, 'not_contains'),

    # operation symbol type
    'getitem':      (composite_ns, 'getitem'),
    'ms_iter':      (composite_ns, 'ms_iter'),
    'ms_next':      (composite_ns, 'ms_next'),
    'hasnext':      (composite_ns, 'hasnext'),

    # undefined type
    SYMBOL_UNDEFINE: (None, 'undefine'),
}

# Operation symbols corresponding to ast grammar
ops_symbol_map = {
    # ast grammar
    ast.Add:        '+',
    ast.Sub:        '-',
    ast.Mult:       '*',
    ast.Div:        '/',
    ast.FloorDiv:   '//',
    ast.Mod:        '%',
    ast.Pow:        '**',
    ast.LShift:     '<<',
    ast.RShift:     '>>',
    ast.BitAnd:     '&',
    ast.BitOr:      '|',
    ast.BitXor:     '^',

    # undefined type
    SYMBOL_UNDEFINE: '',
}

# Escape an object to another object, eg: system function(len,xxx)
# Some space set aside for readability of code
convert_object_map = {
    T.add:          multitype_ops.add,
    T.sub:          multitype_ops.sub,
    T.mul:          multitype_ops.mul,
    T.truediv:      multitype_ops.div,
    T.getitem:      multitype_ops.getitem,
    T.setitem:      multitype_ops.setitem,
    T.floordiv:     multitype_ops.floordiv,
    T.mod:          multitype_ops.mod,
    T.pow:          multitype_ops.pow_,
    T.matmul:       F.matmul,
    T.lshift:       multitype_ops.left_shift,
    T.rshift:       multitype_ops.right_shift,
    T.and_:         multitype_ops.bitwise_and,
    T.or_:          multitype_ops.bitwise_or,
    T.xor:          multitype_ops.bitwise_xor,
    T.pos:          multitype_ops.uadd,
    T.neg:          multitype_ops.negative,
    T.invert:       F.logical_not,
    T.not_:         multitype_ops.logical_not,
    T.eq:           multitype_ops.equal,
    T.ne:           multitype_ops.not_equal,
    T.lt:           multitype_ops.less,
    T.gt:           multitype_ops.greater,
    T.le:           multitype_ops.less_equal,
    T.ge:           multitype_ops.greater_equal,
    T.is_:          F.is_,
    T.is_not:       F.is_not,
    T.contains:     multitype_ops.in_,
    T.not_contains: multitype_ops.not_in_,

    # system function
    T.abs:          M.ms_abs,
    T.round:        M.ms_round,
    T.len:          M.ms_len,
    T.bool_:        M.bool_,
    T.map:          C.Map(),
    T.filter:       M.filter_,
    T.partial:      F.partial,
    T.zip:          C.zip_operation,
    T.enumerate:    M.enumerate_,
    T.isinstance:   Primitive('isinstance'),
    T.max:          M.ms_max,
    T.min:          M.ms_min,
    T.sum:          M.ms_sum,
    T.getattr:      Primitive('getattr'),
    T.hasattr:      M.hasattr,

    # custom define operation
    T.iter:         M.ms_iter,
    T.next:         M.ms_next,
    T.hasnext:      M.hasnext,
    T.MakeTuple:    F.make_tuple,
    T.make_dict:    F.make_dict,
    T.make_list:    F.make_list,
    T.make_slice:   F.make_slice,
    T.range:        F.make_range,
    T.while_cond:   M.while_cond,
    T.mutable:      Primitive('mutable'),

    # lib function
    math.log:       F.scalar_log,

    # user defined
    RowTensorInner: F.make_row_tensor_inner,
    RowTensor:      F.make_row_tensor,
    SparseTensor:   F.make_sparse_tensor,
    COOTensor:      F.make_coo_tensor,
    CSRTensor:      F.make_csr_tensor,
    MapParameter:   F.make_map_parameter
}

if not security.enable_security():
    convert_object_map[T.print] = F.print_

# Convert class object to callable function
convert_class_to_function_map = {
    "class 'list'":  M.list_func,
    "class 'tuple'": M.tuple_func,
    "class 'int'":   M.int_func,
    "class 'float'": M.float_func,
    "class 'bool'":  M.bool_func,
    "class 'str'":   M.str_func
}
