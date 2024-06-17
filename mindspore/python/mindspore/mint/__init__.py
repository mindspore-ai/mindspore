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
from mindspore.ops.extend import *
from mindspore.ops.extend import array_func, math_func, nn_func
from mindspore.mint.nn.functional import *
from mindspore.mint.nn import functional
from mindspore.ops import erf, where, triu
from mindspore.ops.function.math_func import linspace_ext as linspace
from mindspore.ops.function.array_func import full_ext as full
from mindspore.ops.function.array_func import ones_like_ext as ones_like
from mindspore.ops.function.array_func import zeros_like_ext as zeros_like
from mindspore.ops.auto_generate import abs
# 1
from mindspore.ops.function.math_func import divide, div
from mindspore.ops.auto_generate import topk_ext as topk
# 2
from mindspore.ops.function.math_func import sin
# 3
from mindspore.ops.function.clip_func import clamp
# 4

# 5

# 6
from mindspore.ops.auto_generate import stack_ext as stack

# 7

# 8

# 9

# 10
from mindspore.ops.function.math_func import ne
# 11

# 12
from mindspore.ops.function.array_func import repeat_interleave_ext as repeat_interleave
# 13

# 14

# 15
from mindspore.ops.auto_generate import flatten_ext as flatten
# 16
from mindspore.ops.functional import matmul
# 17

# 18
from mindspore.ops.functional import sum
# 19
from mindspore.ops.functional import log
# 20
from mindspore.ops.functional import prod
# 21
from mindspore.ops.functional import mul
# 22

# 23
from mindspore.ops.functional import mean_ext as mean
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
from mindspore.ops.functional import sqrt
# 30

# 31

# 32

# 33
from mindspore.ops.function.array_func import split_ext as split
# 34

# 35
from mindspore.ops.functional import erfinv
# 36

# 37

# 38

# 39

# 40

# 41

# 42
from mindspore.ops.functional import argmax
# 43

# 44
from mindspore.ops.functional import cos
# 45

# 46

# 47

# 48

# 49

# 50
from mindspore.ops.functional import tile
# 51

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
from mindspore.ops.function.math_func import all

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

# 208
from mindspore.ops.function.array_func import eye

# 285
from mindspore.ops.function.array_func import scatter_add_ext as scatter_add

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
    # 4

    # 5

    # 6
    'stack',
    # 7

    # 8

    # 9

    # 10
    'ne',
    # 11

    # 12
    "repeat_interleave",
    # 13

    # 14

    # 15
    'flatten',
    # 16
    'matmul',
    # 17

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
    'mean',
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

    # 31

    # 32

    # 33
    'split',
    # 34

    # 35
    'erfinv',
    # 36

    # 37

    # 38

    # 39

    # 40

    # 41

    # 42
    'argmax',
    # 43

    # 44
    'cos',
    # 45

    # 46

    # 47

    # 48

    # 49

    # 50
    'tile',
    # 51

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

    # 85

    # 86

    # 87

    # 88
    'chunk',
    # 89

    # 90

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

    # 208
    'eye',

    # 285
    'scatter_add',
    # 304

    # 305
    'triu',
]
__all__.extend(array_func.__all__)
__all__.extend(math_func.__all__)
__all__.extend(nn_func.__all__)
__all__.extend(functional.__all__)
