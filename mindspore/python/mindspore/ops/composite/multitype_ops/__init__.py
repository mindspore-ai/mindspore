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

from .add_impl import add
from .sub_impl import sub
from .mul_impl import mul
from .div_impl import div
from .pow_impl import pow_
from .floordiv_impl import floordiv
from .mod_impl import mod
from .getitem_impl import getitem
from .setitem_impl import setitem
from .zeros_like_impl import zeros_like
from .ones_like_impl import ones_like
from .equal_impl import equal
from .not_equal_impl import not_equal
from .less_impl import less
from .less_equal_impl import less_equal
from .greater_impl import greater
from .greater_equal_impl import greater_equal
from .negative_impl import negative
from .logical_and_impl import logical_and
from .logical_or_impl import logical_or
from .logic_not_impl import logical_not
from .uadd_impl import uadd
from .in_impl import in_
from .not_in_impl import not_in_
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
    'in_',
    'not_in_'
]
