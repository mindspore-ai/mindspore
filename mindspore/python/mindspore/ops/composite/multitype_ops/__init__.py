# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Multitype ops"""

from mindspore.ops.composite.multitype_ops.add_impl import add
from mindspore.ops.composite.multitype_ops.sub_impl import sub
from mindspore.ops.composite.multitype_ops.mul_impl import mul
from mindspore.ops.composite.multitype_ops.div_impl import div
from mindspore.ops.composite.multitype_ops.pow_impl import pow_
from mindspore.ops.composite.multitype_ops.floordiv_impl import floordiv
from mindspore.ops.composite.multitype_ops.mod_impl import mod
from mindspore.ops.composite.multitype_ops.getitem_impl import getitem
from mindspore.ops.composite.multitype_ops.setitem_impl import setitem
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.composite.multitype_ops.ones_like_impl import ones_like
from mindspore.ops.composite.multitype_ops.equal_impl import equal
from mindspore.ops.composite.multitype_ops.not_equal_impl import not_equal
from mindspore.ops.composite.multitype_ops.less_impl import less
from mindspore.ops.composite.multitype_ops.less_equal_impl import less_equal
from mindspore.ops.composite.multitype_ops.greater_impl import greater
from mindspore.ops.composite.multitype_ops.greater_equal_impl import greater_equal
from mindspore.ops.composite.multitype_ops.negative_impl import negative
from mindspore.ops.composite.multitype_ops.logical_and_impl import logical_and
from mindspore.ops.composite.multitype_ops.logical_or_impl import logical_or
from mindspore.ops.composite.multitype_ops.logic_not_impl import logical_not
from mindspore.ops.composite.multitype_ops.bitwise_and_impl import bitwise_and
from mindspore.ops.composite.multitype_ops.bitwise_or_impl import bitwise_or
from mindspore.ops.composite.multitype_ops.bitwise_xor_impl import bitwise_xor
from mindspore.ops.composite.multitype_ops.left_shift_impl import left_shift
from mindspore.ops.composite.multitype_ops.right_shift_impl import right_shift
from mindspore.ops.composite.multitype_ops.uadd_impl import uadd
from mindspore.ops.composite.multitype_ops.in_impl import in_
from mindspore.ops.composite.multitype_ops.not_in_impl import not_in_
__all__ = [
    'add',
    'sub',
    'mul',
    'div',
    'pow_',
    'floordiv',
    'mod',
    'uadd',
    'zeros_like',
    'ones_like',
    'equal',
    'not_equal',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
    'negative',
    'getitem',
    'setitem',
    'logical_and',
    'logical_or',
    'logical_not',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'left_shift',
    'right_shift',
    'in_',
    'not_in_'
]
