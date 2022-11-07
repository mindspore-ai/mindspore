#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License == distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

_basic
"""
from __future__ import absolute_import
import te.platform.cce_params as cce


def _get_km_kn_shape(shape_a, shape_b, trans_a, trans_b):
    """get_km_kn_shape"""
    shape_len = len(shape_a)
    if trans_a:
        m_shape = shape_a[shape_len - 1]
        km_shape = shape_a[shape_len - 2]
    else:
        m_shape = shape_a[shape_len - 2]
        km_shape = shape_a[shape_len - 1]

    if trans_b:
        kn_shape = shape_b[shape_len - 1]
        n_shape = shape_b[shape_len - 2]
    else:
        kn_shape = shape_b[shape_len - 2]
        n_shape = shape_b[shape_len - 1]
    return m_shape, km_shape, n_shape, kn_shape


def _check_mn_shape(m_shape, n_shape, km_shape, kn_shape):
    """_check_mn_shape"""
    if m_shape == 1 and n_shape == 1:
        raise RuntimeError("input shape M and N can't both be 1")

    if km_shape != kn_shape:
        raise RuntimeError("reduce axis not same")

    if m_shape % cce.BLOCK_IN != 0 and m_shape != 1:
        raise RuntimeError(
            "input shape M should be 1 or multiple of %d" % cce.BLOCK_IN)

    if m_shape != 1:
        if n_shape == 1 and km_shape % (cce.BLOCK_IN * cce.BLOCK_IN) != 0:
            raise RuntimeError("input shape K1 must be multiple of %d"
                               % (cce.BLOCK_IN * cce.BLOCK_IN))
        if km_shape % cce.BLOCK_REDUCE != 0:
            raise RuntimeError(
                "input shape K1 should be multiple of %d" % cce.BLOCK_IN)
    else:
        if km_shape % (cce.BLOCK_IN * cce.BLOCK_IN) != 0:
            raise RuntimeError("input shape K1 must be multiple of %d"
                               % (cce.BLOCK_IN * cce.BLOCK_IN))


def _check_bias(shape_bias, shape_a, shape_b, m_shape, n_shape):
    """_check_bias"""
    is_gevm = True if shape_a[-2] == 1 or shape_a[-1] == 1 else False
    is_gemv = True if shape_b[-2] == 1 or shape_b[-1] == 1 else False
    if shape_bias:
        if len(shape_bias) == 1:
            if (is_gevm or is_gemv) and shape_bias[0] != m_shape * n_shape:
                raise RuntimeError("broadcast case shape bias for gemv must be equal m*n")
            if shape_bias[0] != n_shape:
                raise RuntimeError("broadcast bias shape must be equal to shape n")
        elif len(shape_bias) == len(shape_a):
            if [i for i in shape_bias[-2:]] != [m_shape, n_shape]:
                raise RuntimeError("non broadcast bias shape must be same as output shape")
        else:
            raise RuntimeError("Unsupported input shape now for batch bias case")


def _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    src_dtype: str
            The data type of input, support "float32", "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
           If True, shape_b == transposed before multiplication

    Returns None
    """
    shape_len = len(shape_a)
    src_dtype = src_dtype.lower()

    check_list = ("float16",)

    if src_dtype not in check_list:
        raise RuntimeError("matmul_cce only support %s while src_dtype == %s"
                           % (",".join(check_list), src_dtype))
    if shape_len != len(shape_b):
        raise RuntimeError("length of a and b are not equal")

    if shape_len != 2:
        raise RuntimeError(
            "length of shape must be 2, more than 2 dimensions must use batch_matmul now!")

    m_shape, km_shape, n_shape, kn_shape = _get_km_kn_shape(shape_a, shape_b, trans_a, trans_b)

    if n_shape % cce.BLOCK_IN != 0 and n_shape != 1:
        raise RuntimeError("input shape N must be 1 or multiple of %d" % cce.BLOCK_IN)
    _check_mn_shape(m_shape, n_shape, km_shape, kn_shape)
    _check_bias(shape_bias, shape_a, shape_b, m_shape, n_shape)


def _get_bias(shape_bias):
    """_get_bias"""
    bias_length = shape_bias[0]
    if bias_length % 16 == 0:
        shb = shape_bias
    else:
        bias_length = (bias_length // 16) * 16 + 16
        shape_bias = []
        shape_bias.append(bias_length)
        shb = shape_bias
    return shb


def _get_input_shape(shape_x):
    """_get_input_shape"""
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = []
    if dim_a % 16 != 0:
        dim_a = (dim_a // 16) * 16 + 16
        res.append(dim_a)
    else:
        res.append(dim_a)

    if dim_b % 16 != 0:
        dim_b = (dim_b // 16) * 16 + 16
        res.append(dim_b)
    else:
        res.append(dim_b)
    return res
