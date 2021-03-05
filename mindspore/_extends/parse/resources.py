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
"""Resources for ast tree parse."""
import ast
import math

from mindspore import RowTensor, SparseTensor
from mindspore.ops import functional as F, composite as C
from mindspore.ops.composite import multitype_ops
from . import standard_method as M
from . import trope as T
from .namespace import CellNamespace

# namespace define
functional_ns = CellNamespace('mindspore.ops.functional')
composite_ns = CellNamespace('mindspore.ops.composite')
trope_ns = CellNamespace('mindspore._extends.parse.trope')

NO_IMPLEMENT = None         # not implemented
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
    ast.And:        (trope_ns, 'and_'),
    ast.Or:         (trope_ns, 'or_'),
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
    T.lshift:       NO_IMPLEMENT,
    T.rshift:       NO_IMPLEMENT,
    T.and_:         multitype_ops.logical_and,
    T.or_:          multitype_ops.logical_or,
    T.xor:          NO_IMPLEMENT,
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
    T.len:          M.ms_len,
    T.bool_:        M.bool_,
    T.map:          C.Map(),
    T.partial:      F.partial,
    T.zip:          C.zip_operation,
    T.print:        F.print_,
    T.enumerate:    M.enumerate_,
    T.isinstance:   M.isinstance_,

    # custom define operation
    T.iter:         M.ms_iter,
    T.next:         M.ms_next,
    T.hasnext:      M.hasnext,

    T.MakeTuple:   F.make_tuple,
    T.make_dict:    F.make_dict,
    T.make_list:    F.make_list,
    T.make_slice:   F.make_slice,
    T.range:        F.make_range,
    T.while_cond:   M.while_cond,
    # lib function
    math.floor:     NO_IMPLEMENT,
    math.trunc:     NO_IMPLEMENT,
    math.exp:       NO_IMPLEMENT,
    math.log:       F.scalar_log,
    math.sin:       NO_IMPLEMENT,
    math.cos:       NO_IMPLEMENT,
    math.tan:       NO_IMPLEMENT,

    # user defined
    RowTensor:  F.make_row_tensor,
    SparseTensor:   F.make_sparse_tensor,
}
