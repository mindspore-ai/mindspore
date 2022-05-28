# Copyright 2022 Huawei Technologies Co., Ltd
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
Function operator.

A collection of function to build neural networks or to compute functions.
"""

from . import (
    array_func,
    parameter_func,
    math_func,
    nn_func,
    linalg_func,
    clip_func,
)
from .array_func import (
    unique,
    eye,
    matrix_band_part,
    padding,
    fill,
    fill_,
    tile,
    size,
    ones,
    ones_like,
    shape,
    shape_,
    ger,
    dyn_shape,
    rank,
    reshape,
    reshape_,
    tensor_slice,
    slice,
    scalar_to_array,
    scalar_to_tensor,
    tuple_to_array,
    expand_dims,
    transpose,
    scatter_nd,
    scatter_nd_add,
    scatter_nd_sub,
    gather,
    gather_d,
    gather_nd,
    scalar_cast,
    masked_fill,
    tensor_scatter_add,
    tensor_scatter_sub,
    tensor_scatter_mul,
    unique_consecutive,
    tensor_scatter_div,
    scatter_max,
    scatter_min,
    nonzero,
    space_to_batch_nd,
    range,
    select,
    one_hot,
    matrix_diag,
    diag,
    masked_select,
    meshgrid,
    fills,
)
from .parameter_func import (
    assign,
    assign_add,
    assign_sub,
    index_add,
)
from .math_func import (
    addn,
    absolute,
    abs,
    tensor_add,
    add,
    neg_tensor,
    neg,
    tensor_lt,
    less,
    tensor_le,
    le,
    lerp,
    lp_norm,
    round,
    tensor_gt,
    gt,
    tensor_ge,
    ge,
    tensor_sub,
    sub,
    tensor_mul,
    mul,
    tensor_div,
    div,
    tensor_floordiv,
    floor_div,
    floordiv,
    tensor_pow,
    pow,
    pows,
    tensor_mod,
    floor_mod,
    floormod,
    tensor_exp,
    exp,
    tensor_expm1,
    expm1,
    equal,
    not_equal,
    ne,
    isfinite,
    isnan,
    same_type_shape,
    log,
    log_matrix_determinant,
    matrix_determinant,
    maximum,
    logaddexp,
    logaddexp2,
    mv,
    inv,
    invert,
    minimum,
    floor,
    logical_not,
    logical_or,
    logical_and,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    atan2,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    erf,
    erfc,
    cdist,
    bessel_i0,
    bessel_i0e,
    bessel_j0,
    bessel_j1,
    bessel_k0,
    bessel_k0e,
    bessel_y0,
    bessel_y1,
    bessel_i1,
    bessel_i1e,
    bessel_k1,
    bessel_k1e,
    exp2,
    deg2rad,
)
from .nn_func import (
    fast_gelu,
    hardshrink,
    softsign,
    pdist,
)
from .linalg_func import (
    svd,
)
from .clip_func import (
    clip_by_norm,
)

__all__ = []
__all__.extend(array_func.__all__)
__all__.extend(parameter_func.__all__)
__all__.extend(math_func.__all__)
__all__.extend(nn_func.__all__)
__all__.extend(linalg_func.__all__)
__all__.extend(clip_func.__all__)
__all__.sort()
