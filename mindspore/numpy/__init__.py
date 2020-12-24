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
    - logic_ops.py defines all the logical operations on tensors.
    - dtypes.py defines all the mindspore.numpy dtypes (mainly redirected from mindspore)
"""

from .array_ops import (transpose, expand_dims, squeeze, rollaxis, swapaxes, reshape,
                        ravel, concatenate, where, atleast_1d, atleast_2d, atleast_3d,
                        column_stack, hstack, dstack, vstack, stack, unique, moveaxis,
                        tile, broadcast_to, broadcast_arrays, roll, append, split, vsplit,
                        flip, flipud, fliplr, hsplit, dsplit, take_along_axis, take, repeat,
                        rot90, select, array_split)
from .array_creations import copy_ as copy
from .array_creations import (array, asarray, asfarray, ones, zeros, full, arange,
                              linspace, logspace, eye, identity, empty, empty_like,
                              ones_like, zeros_like, full_like, diagonal, tril, triu,
                              tri, trace, meshgrid, mgrid, ogrid, diagflat,
                              diag, diag_indices, ix_, indices, geomspace, vander)
from .dtypes import (int_, int8, int16, int32, int64, uint, uint8, uint16,
                     uint32, uint64, float_, float16, float32, float64, bool_, inf, nan,
                     numeric_types, PINF, NINF)
from .math_ops import (mean, inner, add, subtract, multiply, divide, true_divide, power,
                       dot, outer, tensordot, absolute, std, var, average, minimum,
                       matmul, square, sqrt, reciprocal, log, maximum, heaviside, amax, amin,
                       hypot, float_power, floor, ptp, deg2rad, rad2deg, count_nonzero,
                       positive, negative, clip, floor_divide, remainder, fix, fmod, trunc,
                       exp, expm1, exp2, kron, promote_types, divmod_, diff, cbrt,
                       cross, ceil, trapz, gcd, lcm, convolve, log1p, logaddexp, log2,
                       logaddexp2, log10, ediff1d, nansum, nanmean, nanvar, nanstd, cumsum, nancumsum,
                       sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arcsinh, arccosh,
                       arctanh, arctan2, cov)
from .logic_ops import (not_equal, less_equal, less, greater_equal, greater, equal, isfinite,
                        isnan, isinf, isposinf, isneginf, isscalar, logical_and, logical_not,
                        logical_or, logical_xor, in1d, isin, isclose)

mod = remainder
fabs = absolute
divmod = divmod_ # pylint: disable=redefined-builtin
abs = absolute # pylint: disable=redefined-builtin
max = amax # pylint: disable=redefined-builtin
min = amin # pylint: disable=redefined-builtin


array_ops_module = ['transpose', 'expand_dims', 'squeeze', 'rollaxis', 'swapaxes', 'reshape',
                    'ravel', 'concatenate', 'where', 'atleast_1d', 'atleast_2d', 'atleast_3d',
                    'column_stack', 'hstack', 'dstack', 'vstack', 'stack', 'unique', 'moveaxis',
                    'tile', 'broadcast_to', 'broadcast_arrays', 'append', 'roll', 'split', 'vsplit',
                    'flip', 'flipud', 'fliplr', 'hsplit', 'dsplit', 'take_along_axis', 'take',
                    'repeat', 'rot90', 'select', 'array_split']

array_creations_module = ['array', 'asarray', 'asfarray', 'ones', 'zeros', 'full', 'arange',
                          'linspace', 'logspace', 'eye', 'identity', 'empty', 'empty_like',
                          'ones_like', 'zeros_like', 'full_like', 'diagonal', 'tril', 'triu',
                          'tri', 'trace', 'meshgrid', 'mgrid', 'ogrid', 'diagflat', 'diag',
                          'diag_indices', 'ix_', 'indices', 'geomspace', 'vander']

math_module = ['mean', 'inner', 'add', 'subtract', 'multiply', 'divide', 'true_divide', 'power',
               'dot', 'outer', 'tensordot', 'absolute', 'std', 'var', 'average', 'not_equal',
               'minimum', 'matmul', 'square', 'sqrt', 'reciprocal', 'log', 'maximum',
               'heaviside', 'amax', 'amin', 'hypot', 'float_power', 'floor', 'ptp', 'deg2rad',
               'rad2deg', 'count_nonzero', 'positive', 'negative', 'clip', 'floor_divide',
               'remainder', 'mod', 'fix', 'fmod', 'trunc', 'exp', 'expm1', 'fabs', 'exp2', 'kron',
               'promote_types', 'divmod', 'diff', 'cbrt', 'cross', 'ceil', 'trapz',
               'abs', 'max', 'min', 'gcd', 'lcm', 'log1p', 'logaddexp', 'log2', 'logaddexp2', 'log10',
               'convolve', 'ediff1d', 'nansum', 'nanmean', 'nanvar', 'nanstd', 'cumsum',
               'nancumsum', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
               'arcsinh', 'arccosh', 'arctanh', 'arctan2', 'cov']

logic_module = ['not_equal', 'less_equal', 'less', 'greater_equal', 'greater', 'equal', 'isfinite',
                'isnan', 'isinf', 'isposinf', 'isneginf', 'isscalar', 'logical_and', 'logical_not',
                'logical_or', 'logical_xor', 'in1d', 'isin', 'isclose']

__all__ = array_ops_module + array_creations_module + math_module + logic_module + numeric_types

__all__.sort()
