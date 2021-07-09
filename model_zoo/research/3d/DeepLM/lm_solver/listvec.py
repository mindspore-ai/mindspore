# Copyright 2021 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""list-vector operations."""
import mindspore.ops.operations as P
import mindspore.ops as ops
from mindspore.ops import functional as F

f_arg_max_with_value = P.ArgMaxWithValue()
f_zeros = P.Zeros()
f_batchmatmul = P.BatchMatMul()
f_matrix_inverse = P.MatrixInverse()


def list_norm(listvec):
    n = 0
    for vec in listvec:
        n += F.reduce_sum(F.mul(vec, vec))
    return F.sqrt(n)


def list_max_norm(listvec):
    """max norm in the list"""
    n = 0
    for vec in listvec:
        try:
            if vec.shape:
                m = f_arg_max_with_value(f_arg_max_with_value(F.absolute(vec))[1])[1]
                if m > n:
                    n = m
        except AttributeError:
            continue
    return n


def list_clamp(listvec, minVal, maxVal):
    for _, i in enumerate(range(len(listvec))):
        listvec[i] = ops.clip_by_value(listvec[i], minVal, maxVal)


def list_invert(listvec):
    for _, i in enumerate(range(len(listvec))):
        listvec[i] = f_matrix_inverse(listvec[i])


def list_zero(variables):
    zeros = []
    for v in variables:
        vplain = F.reshape(v, (v.shape[0], -1))
        zeros.append(f_zeros(vplain.shape, v.dtype))
    return zeros


def list_right_multiply(p, r, res):
    for _, varid in enumerate(range(len(r))):
        res[varid] = F.reshape(f_batchmatmul(p[varid], F.reshape(r[varid], (r[varid].shape[0], -1, 1))),
                               (r[varid].shape))


def list_dot(a, b):
    res = 0
    for _, i in enumerate(range(len(a))):
        res += F.reduce_sum(F.mul(a[i], b[i]))
    return res
