# Copyright 2024 Huawei Technologies Co., Ltd
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
"""mint module."""
from __future__ import absolute_import
from mindspore.ops.extend import gather, max, min, one_hot, bmm, conv2d
from mindspore.mint.nn.functional import *
from mindspore.mint.nn import functional
from mindspore.ops import erf, where, tril, triu
from mindspore.ops.function.math_func import linspace_ext as linspace
from mindspore.ops.function.array_func import full_ext as full
from mindspore.ops.function.array_func import ones_like_ext as ones_like
from mindspore.ops.function.array_func import zeros_like_ext as zeros_like
from mindspore.ops.function.array_func import unique_ext as unique
from mindspore.ops.auto_generate import abs
# 1
from mindspore.ops.function.math_func import divide, div
from mindspore.ops.function.array_func import topk_ext as topk
# 2
from mindspore.ops.function.math_func import sin
# 3
from mindspore.ops.function.math_func import cross_ext as cross
from mindspore.ops.function.clip_func import clamp
# 4
from mindspore.ops.function.math_func import xlogy_ext as xlogy
# 5
from mindspore.ops.auto_generate import cumsum_ext as cumsum
# 6
from mindspore.ops.auto_generate import stack_ext as stack

# 7
from mindspore.ops.auto_generate import ones as ones_ex
from mindspore.ops.auto_generate import zeros as zeros_ex
# 8

# 9
from mindspore.ops.auto_generate import masked_select_ext as masked_select
# 10
from mindspore.ops.function.math_func import ne
# 11

# 12
from mindspore.ops.function.array_func import repeat_interleave_ext as repeat_interleave
# 13
from mindspore.ops.auto_generate import flip as flip_ex

# 14

# 15
from mindspore.ops.auto_generate import flatten_ext as flatten
# 16
from mindspore.ops.functional import matmul
# 17
from mindspore.ops.auto_generate import mean as mean_ex

# 18
from mindspore.ops.functional import sum
# 19
from mindspore.ops.functional import log
# 20
from mindspore.ops import prod
# 21
from mindspore.ops.functional import mul
# 22

# 23

# 24

# 25
from mindspore.ops.functional import greater, gt
# 26
from mindspore.ops.functional import eq
# 27
from mindspore.ops.functional import reciprocal
# 28
from mindspore.ops.functional import exp
# 29
from mindspore.ops.functional import sqrt as sqrt_ex

# 30
from mindspore.ops.functional import searchsorted
# 31

# 32
from mindspore.ops.extend import sub as sub_ex

# 33
from mindspore.ops.function.array_func import split_ext as split
# 34

# 35
from mindspore.ops.functional import erfinv
# 36

# 37
from mindspore.ops.function.array_func import nonzero
# 38

# 39

# 40
from mindspore.ops.functional import any as any_ex

# 41
from mindspore.ops.extend import add as add_ex

# 42
from mindspore.ops.functional import argmax
# 43
from mindspore.ops.functional import cat as cat_ex

# 44
from mindspore.ops.functional import cos
# 45

# 46
from mindspore.ops.function.math_func import bitwise_and_ext as bitwise_and
# 47
from mindspore.ops.function.math_func import bitwise_or_ext as bitwise_or
# 48
from mindspore.ops.function.math_func import bitwise_xor_ext as bitwise_xor
# 49

# 50
from mindspore.ops.functional import tile
# 51
from mindspore.ops.functional import permute as permute_ex

# 52

# 53

# 54
from mindspore.ops import normal_ext as normal
# 55

# 56

# 57
from mindspore.ops.functional import broadcast_to
# 58

# 59
from mindspore.ops.functional import square
# 60
from mindspore.ops.function.math_func import all, cummax, cummin
# 61
from mindspore.ops.functional import rsqrt
# 62
from mindspore.ops.functional import maximum
# 63
from mindspore.ops.functional import minimum
# 64

# 65
from mindspore.ops.functional import logical_and
# 66
from mindspore.ops.functional import logical_not
# 67
from mindspore.ops.functional import logical_or
# 68

# 69
from mindspore.ops.functional import less_equal, le
# 70
from mindspore.ops.functional import negative, neg
# 71
from mindspore.ops.functional import isfinite
# 72

# 73
from mindspore.ops.functional import ceil
# 74
from mindspore.ops.function.array_func import sort_ext as sort
# 75
from mindspore.ops.functional import less, lt
# 76
from mindspore.ops.functional import pow
# 77

# 78
from mindspore.ops.function import arange_ext as arange
# 79

# 80

# 81
from mindspore.ops.function.array_func import index_select_ext as index_select
# 82

# 83

# 84

# 85

# 86

# 87

# 88
from mindspore.ops.function.array_func import chunk_ext as chunk
# 89

# 90

# 91

# 92

# 93

# 94
from mindspore.ops.function.math_func import tanh
# 95

# 96

# 97

# 98

# 99

# 100

# 102
from mindspore.ops.extend import baddbmm as baddbmm_ex

# 157
from mindspore.ops.function.array_func import scatter

# 232
from mindspore.ops.function.math_func import isclose

# 275
from mindspore.ops.function.math_func import remainder_ext as remainder

# 285
from mindspore.ops.function.array_func import scatter_add_ext as scatter_add

# 289
from mindspore.ops.auto_generate import sign


def add(input, other, *, alpha=1):
    return add_ex(input, other, alpha)


def any(input, dim=None, keepdim=False):
    return any_ex(input, dim, keepdim)


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return baddbmm_ex(input, batch1, batch2, beta, alpha)


def cat(tensors, dim=0):
    return cat_ex(tensors, dim)


def flip(input, dims):
    return flip_ex(input, dims)


def mean(input, dim=None, keepdim=False, *, dtype=None):
    return mean_ex(input, axis=dim, keep_dims=keepdim, dtype=dtype)


def ones(size, *, dtype=None):
    return ones_ex(size, dtype)


def permute(input, dims):
    return permute_ex(input, dims)


def sqrt(input):
    return sqrt_ex(input)


def sub(input, other, *, alpha=1):
    return sub_ex(input, other, alpha)


def zeros(size, *, dtype=None):
    return zeros_ex(size, dtype)


__all__ = [
    'full',
    'ones_like',
    'zeros_like',
    'abs',
    'erf',
    'where',
    'linspace',
    # 1
    'div',
    'divide',
    'topk',
    # 2
    'sin',
    # 3
    'clamp',
    'xlogy',
    # 4

    # 5
    'cumsum',
    # 6
    'stack',
    # 7
    'zeros',
    # 8

    # 9

    # 10
    'ne',
    # 11

    # 12
    "repeat_interleave",
    # 13
    'flip',
    # 14

    # 15
    'flatten',
    # 16
    'matmul',
    # 17
    'mean',
    # 18
    'sum',
    # 19
    'log',
    # 20
    'prod',
    # 21
    'mul',
    # 22

    # 23

    # 24

    # 25
    'greater',
    'gt',
    # 26
    'eq',
    # 27
    'reciprocal',
    # 28
    'exp',
    # 29
    'sqrt',
    # 30
    'searchsorted',
    # 31

    # 32
    'sub',
    # 33
    'split',
    # 34

    # 35
    'erfinv',
    # 36

    # 37
    'nonzero',
    # 38

    # 39

    # 40
    'any',
    # 41
    'add',
    # 42
    'argmax',
    # 43
    'cat',
    # 44
    'cos',
    # 45

    # 46
    'bitwise_and',
    # 47
    'bitwise_or',
    # 48
    'bitwise_xor',
    # 49

    # 50
    'tile',
    # 51
    'permute',
    # 52

    # 53

    # 54
    'normal',
    # 55

    # 56

    # 57
    'broadcast_to',
    # 58

    # 59
    'square',
    # 60
    'all',
    # 61
    'rsqrt',
    # 62
    'maximum',
    # 63
    'minimum',
    # 64

    # 65
    'logical_and',
    # 66
    'logical_not',
    # 67
    'logical_or',
    # 68

    # 69
    'less_equal',
    'le',
    # 70
    'negative',
    'neg',
    # 71
    'isfinite',
    # 72

    # 73
    'ceil',
    # 74
    'sort',
    # 75
    'less',
    'lt',
    # 76
    'pow',
    # 77

    # 78
    'arange',

    # 79

    # 80

    # 81
    'index_select',
    # 82

    # 83

    # 84

    'masked_select',

    # 86

    # 87

    # 88
    'chunk',
    # 89

    # 90
    'cross',
    # 91

    # 92

    # 93

    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100

    # 102
    'baddbmm',
    # 157
    'scatter',
    # 232
    'isclose',
    # 275
    'remainder',
    # 285
    'scatter_add',
    # 289
    'sign',
    # 304
    'tril',
    # 305
    'triu',
    'gather',
    'max',
    'min',
    'bmm'
]
__all__.extend(functional.__all__)
