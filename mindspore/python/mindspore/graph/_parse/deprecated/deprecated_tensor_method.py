# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""Deprecated Tensor method"""

deprecated_tensor_method_map = {
    # 1 to

    # 2 masked_fill

    # 3 abs

    # 4 __abs__

    # 5 add
    "add": "deprecated_tensor_add",
    "add_": "deprecated_tensor_add_",
    # 6 all
    "all": "tensor_all",
    # 7 allclose
    "allclose": "tensor_allclose",
    # 8 any
    "any": "reduce_tensor_any",
    # 9 arctan2
    "arctan2": "tensor_arctan2",
    # 10 argmax
    "argmax": "deprecated_tensor_argmax",
    # 11 argmin
    "argmin": "deprecated_tensor_argmin",
    # 12 argsort
    "argsort": "deprecated_tensor_argsort",
    # 13 atan2
    "atan2": "tensor_atan2",
    # 14 bfloat16

    # 15 bmm

    # 16 bool

    # 17 broadcast_to

    # 18 byte

    # 19 ceil

    # 20 chunk
    "chunk": "deprecated_tensor_chunk",
    # 21 clamp

    # 22 clip

    # 23 cos

    # 24 cumprod

    # 25 cumsum
    "cumsum": "deprecated_tensor_cumsum",
    # 26 dim

    # 27 div
    "div": "tensor_div",
    # 28 divide

    # 29 eq

    # 30 erf

    # 31 exp

    # 32 expand

    # 33 expand_as
    "expand_as": "deprecated_tensor_expand_as",
    # 34 flatten
    "flatten": "deprecated_tensor_flatten",
    # 35 flip

    # 36 float

    # 37 floor

    # 38 gather
    "gather": "deprecated_tensor_gather",
    # 39 greater

    # 40 greater_equal
    "greater_equal": "deprecated_tensor_greater_equal",
    # 41 gt

    # 42 half

    # 43 index_put

    # 44 index_select
    "index_select": "deprecated_tensor_index_select",
    # 45 int

    # 46 inverse
    "inverse": "deprecated_tensor_inverse",
    # 47 is_contiguous

    # 48 isclose
    "isclose": "deprecated_tensor_isclose",
    # 49 isfinite

    # 50 isnan

    # 51 item

    # 52 le

    # 53 less

    # 54 less_equal

    # 55 log

    # 56 log2
    "log2": "tensor_log2",
    # 57 logical_and
    "logical_and": "tensor_logical_and",
    # 58 logical_not

    # 59 logical_or
    "logical_or": "tensor_logical_or",
    # 60 long

    # 61 lt

    # 62 masked_fill

    # 63 masked_select

    # 64 matmul
    "matmul": "deprecated_tensor_matmul",
    # 65 max
    "max": "deprecated_tensor_max",
    # 66 maximum

    # 67 mean
    "mean": "deprecated_tensor_mean",
    # 68 min
    "min": "deprecated_tensor_min",
    # 69 minimum

    # 70 mul
    "mul": "deprecated_tensor_mul",

    # 71 nan_to_num

    # 72 narrow
    "narrow": "deprecated_tensor_narrow",
    # 73 ne

    # 74 neg

    # 75 negative

    # 76 nonzero

    # 77 norm

    # 78 numel

    # 79 numpy

    # 80 outer
    "outer": "deprecated_tensor_outer",
    # 81 permute
    "permute": "deprecated_tensor_permute",
    # 82 pow
    "pow": "deprecated_tensor_pow",
    # 83 prod
    "prod": "deprecated_tensor_prod",
    # 84 reciprocal

    # 85 remainder
    "remainder": "deprecated_tensor_remainder",

    # 86 repeat

    # 87 repeat_interleave
    "repeat_interleave": "deprecated_tensor_repeat_interleave",
    # 88 reshape

    # 89 round

    # 90 rsqrt

    # 91 scatter
    "scatter": "deprecated_tensor_scatter",

    # 92 scatter_add
    "scatter_add": "deprecated_tensor_scatter_add",
    # 93 select
    "select": "deprecated_tensor_select",
    # 94 sigmoid

    # 95 sin

    # 96 size

    # 97 sort
    "sort": "deprecated_tensor_sort",
    # 98 split
    "split": "deprecated_tensor_split",
    # 99 sqrt

    # 100 square

    # 101 squeeze
    "squeeze": "tensor_squeeze",
    # 102 std
    "std": "deprecated_tensor_std",
    # 103 sub
    "sub": "deprecated_tensor_sub",
    # 104 sum
    "sum": "deprecated_tensor_sum",
    # 105 swapaxes

    # 106 t
    "t": "deprecated_tensor_t",
    # 107 tanh

    # 108 tile
    "tile": "deprecated_tensor_tile",
    # 109 tolist

    # 110 topk
    "topk": "deprecated_tensor_topk",
    # 111 transpose
    "transpose": "deprecated_tensor_transpose",
    # 112 tril
    "tril": "deprecated_tensor_tril",
    # 113 trunc

    # 114 type

    # 115 type_as
    "type_as": "deprecated_tensor_type_as",
    # 116 unbind
    "unbind": "deprecated_tensor_unbind",
    # 117 unfold

    # 118 unique
    "unique": "deprecated_tensor_unique",
    # 119 unsqeeze

    # 120 view

    # 121 contiguous

    # 122 where
    "where": "deprecated_tensor_where",
    # 123 div_

    # 124 fill_

    # 125 floor_

    # 126 masked_fill_

    # 127 mul_

    # 128 normal_

    # 129 requires_grad_

    # 130 sub_
    "sub_": "deprecated_tensor_sub_",
    # 131 uniform_

    # 132 absolute

    # 133 bincount
    "bincount": "tensor_bincount",
    "roll": "tensor_roll",
    # 134 diff

    # 135 double

    # 136 lcm

    # 137 mm
    "mm": "deprecated_tensor_mm",
    # 138 ravel

    # 139 nelement

    # 140 stride

    # 141 indices

    # 142 view_as
    "view_as": "deprecated_tensor_view_as",
    # 143 values

    # 144 index_copy

    # 145 element_size

    # 146 gcd

    # 147 isinf

    # 148 not_equal

    # 149 triu

    # 150 __eq__

    # 151

    # 152
    'median': 'deprecated_tensor_median',

    # 153 acos, arccos; acosh, arccosh, asin, arcsin; asinh, arcsinh; atan, arctan; dot
    "acos": "deprecated_tensor_acos",
    "arccos": "deprecated_tensor_arccos",
    "acosh": "deprecated_tensor_acosh",
    "arccosh": "deprecated_tensor_arccosh",
    "asin": "deprecated_tensor_asin",
    "arcsin": "deprecated_tensor_arcsin",
    "asinh": "deprecated_tensor_asinh",
    "arcsinh": "deprecated_tensor_arcsinh",
    "atan": "deprecated_tensor_atan",
    "arctan": "deprecated_tensor_arctan",
    "dot": "deprecated_tensor_dot",

    # 153
    "logsumexp": "deprecated_tensor_logsumexp",

    # 154

    # 155
    "isneginf": "deprecated_tensor_isneginf",
    # 156

    # 157
    "logaddexp": "deprecated_tensor_logaddexp",

    "logaddexp2": "deprecated_tensor_logaddexp2",

    "xlogy": "tensor_xlogy",

    # 158
    "unsqueeze": "deprecated_tensor_unsqueeze",
    # 159 histc
    "histc": "tensor_histc",

    # 160 frac
    "frac": "tensor_frac",

    # 161
    "fmod": "deprecated_tensor_fmod",
    "bitwise_or": "deprecated_bitwise_or",
    "bitwise_and": "deprecated_bitwise_and",
    "bitwise_xor": "deprecated_bitwise_xor",
    "baddbmm": "deprecated_baddbmm",

    # 162 log10
    "log10": "tensor_log10",
    # 732
    "take": "deprecated_tensor_take",

    # 186
    "addcdiv": "deprecated_tensor_addcdiv",

    # 501
    "addbmm": "deprecated_tensor_addbmm",
    # 931
    "nansum": "deprecated_tensor_nansum",
    # 502
    "addmm": "deprecated_tensor_addmm",
    # 790 addmv
    "addmv": "deprecated_tensor_addmv",
    # 846
    "count_nonzero": "deprecated_tensor_count_nonzero",
    # 1028
    "var": "deprecated_tensor_var",
    # 1029
    "real": "tensor_real",
    # 1030
    "imag": "tensor_imag",
}
