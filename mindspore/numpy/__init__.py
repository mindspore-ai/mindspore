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
"""
Numpy-like interfaces in mindspore.

Examples:
    >>> import mindspore.numpy as np

Note:
    - array_ops.py defines all the array operation interfaces.
    - array_creations.py defines all the array generation interfaces.
    - math_ops.py defines all the math operations on tensors.
    - dtypes.py defines all the mindspore.numpy dtypes (mainly redirected from mindspore)
"""

from .array_ops import (transpose, expand_dims, squeeze, rollaxis, swapaxes, reshape,
                        ravel, concatenate, where, atleast_1d, atleast_2d, atleast_3d,
                        column_stack, hstack, dstack, vstack, stack, unique)
from .array_creations import copy_ as copy
from .array_creations import (array, asarray, asfarray, ones, zeros, full, arange,
                              linspace, logspace, eye, identity, empty, empty_like,
                              ones_like, zeros_like, full_like, diagonal, tril, triu,
                              tri, trace)
from .dtypes import (int_, int8, int16, int32, int64, uint, uint8, uint16,
                     uint32, uint64, float_, float16, float32, float64, bool_, inf,
                     numeric_types)
from .math_ops import (mean, inner, add, subtract, multiply, divide, power,
                       dot, outer, tensordot, absolute)


array_ops_module = ['transpose', 'expand_dims', 'squeeze', 'rollaxis', 'swapaxes', 'reshape',
                    'ravel', 'concatenate', 'where', 'atleast_1d', 'atleast_2d', 'atleast_3d',
                    'column_stack', 'hstack', 'dstack', 'vstack', 'stack', 'unique']

array_creations_module = ['array', 'asarray', 'asfarray', 'ones', 'zeros', 'full', 'arange',
                          'linspace', 'logspace', 'eye', 'identity', 'empty', 'empty_like',
                          'ones_like', 'zeros_like', 'full_like', 'diagonal', 'tril', 'triu',
                          'tri', 'trace']

math_module = ['mean', 'inner', 'add', 'subtract', 'multiply', 'divide', 'power',
               'dot', 'outer', 'tensordot', 'absolute']

__all__ = array_ops_module + array_creations_module + math_module + numeric_types

__all__.sort()
