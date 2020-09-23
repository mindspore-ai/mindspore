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

matmul
"""
from __future__ import absolute_import
from impl.matmul_vector import matmul_vector_cce
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
import te.lang.cce
import te.platform.cce_params as cce
from te import tvm
from topi import generic
from topi.cce import util

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)

matmul_cube_op_info = TBERegOp("CusMatMulCube") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmulcube.so") \
    .compute_cost(10) \
    .kernel_name("CusMatMulCube") \
    .partial_flag(True) \
    .attr("transpose_a", "required", "bool", "all") \
    .attr("transpose_b", "required", "bool", "all") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_Default, DataType.F32_FracNZ) \
    .get_op_info()


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
    k_block_size = cce.BLOCK_REDUCE

    check_list = ("float16")

    if src_dtype not in check_list:
        raise RuntimeError("matmul_cce only support %s while src_dtype == %s"
                           % (",".join(check_list), src_dtype))
    if shape_len != len(shape_b):
        raise RuntimeError("length of a and b are not equal")

    if shape_len != 2:
        raise RuntimeError(
            "length of shape must be 2, more than 2 dimensions should use batch_matmul now!")

    is_gevm = True if shape_a[-2] == 1 or shape_a[-1] == 1 else False
    is_gemv = True if shape_b[-2] == 1 or shape_b[-1] == 1 else False

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

    if m_shape == 1:
        if n_shape == 1:
            raise RuntimeError("input shape M and N can't both be 1")

    if km_shape != kn_shape:
        raise RuntimeError("reduce axis not same")

    if m_shape % cce.BLOCK_IN != 0 and m_shape != 1:
        raise RuntimeError(
            "input shape M should be 1 or multiple of %d" % cce.BLOCK_IN)

    if m_shape != 1:
        if n_shape == 1:
            if km_shape % (cce.BLOCK_IN * cce.BLOCK_IN) != 0:
                raise RuntimeError("input shape K1 should be multiple of %d"
                                   % (cce.BLOCK_IN * cce.BLOCK_IN))
        elif km_shape % k_block_size != 0:
            raise RuntimeError(
                "input shape K1 should be multiple of %d" % cce.BLOCK_IN)
    else:
        if km_shape % (cce.BLOCK_IN * cce.BLOCK_IN) != 0:
            raise RuntimeError("input shape K1 should be multiple of %d"
                               % (cce.BLOCK_IN * cce.BLOCK_IN))

    if n_shape % cce.BLOCK_IN != 0 and n_shape != 1:
        raise RuntimeError("input shape N should be 1 or multiple of %d" % cce.BLOCK_IN)

    if shape_bias:
        if len(shape_bias) == 1:
            if is_gevm or is_gemv:
                if shape_bias[0] != m_shape * n_shape:
                    raise RuntimeError("broadcast case shape bias for gemv must be equal m*n")
            else:
                if shape_bias[0] != n_shape:
                    raise RuntimeError("broadcast bias shape must be equal to shape n")
        elif len(shape_bias) == shape_len:
            if [i for i in shape_bias[-2:]] != [m_shape, n_shape]:
                raise RuntimeError("non broadcast bias shape must be same as output shape")
        else:
            raise RuntimeError("unsupported input shape now for batch bias case")


def _get_bias(shape_bias):
    """_get_bias"""
    bias_length = shape_bias[0]
    if bias_length % 16 == 0:
        return shape_bias
    bias_length = (bias_length // 16) * 16 + 16
    shape_bias = []
    shape_bias.append(bias_length)
    return shape_bias


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


def check_supported(input_x1, input_x2, bias=None, output_y={}, trans_a=False, trans_b=False, kernel_name="matmulcube"):
    """check_supported"""
    shape_a = input_x1.get("shape")
    shape_b = input_x2.get("shape")
    print("shape_a: ", shape_a)
    print("shape_b: ", shape_b)
    src_dtype = input_x1.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)
    util.check_shape_size(shape_a, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_b, SHAPE_SIZE_LIMIT)
    try:
        trans_a_f = bool(1 - trans_a)
        if src_dtype in ("float32", "int32"):
            if len(shape_a) != 2 and len(shape_b) != 2:
                return False
            if trans_b:
                if shape_b[0] == 1:
                    return False
            else:
                if shape_b[1] == 1:
                    return False
            if trans_a:
                if trans_b:
                    if shape_a[0] != shape_b[1]:
                        return False
                elif shape_a[0] != shape_b[0]:
                    return False
            elif trans_b:
                if shape_a[1] != shape_b[1]:
                    return False
            elif shape_a[1] != shape_b[0]:
                return False

            if trans_a_f and trans_b and shape_b[1] == 1:
                return False

        if src_dtype == "float16":
            if len(shape_a) != 2 and len(shape_b) != 2:
                return False

            if trans_a:
                m_shape = shape_a[1]
                k_shape = shape_a[0]
            else:
                m_shape = shape_a[0]
                k_shape = shape_a[1]

            if trans_b:
                n_shape = shape_b[0]
                k_b_shape = shape_b[1]
            else:
                n_shape = shape_b[1]
                k_b_shape = shape_b[0]

            if k_shape != k_b_shape:
                return False

            if m_shape == 1 or n_shape == 1:
                if k_shape % 256 != 0:
                    return False

    except RuntimeError as e:
        print(e)
        return False

    return True


@op_info_register(matmul_cube_op_info)
def CusMatMulCube(input_x1, input_x2, bias=None, output_y={}, trans_a=False, trans_b=False, kernel_name="matmulcube"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    src_dtype: str
            The data type of input, support "float32", "float16"
    dst_dtype: str
            The data type of output, support "float32", "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
            If True, shape_b == transposed before multiplication
    is_fractal: bool
            If True, the input data format of a and b must be fractal format
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND

    Returns
    -------
    None
    """
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")

    if shape_a is not None:
        if len(shape_a) < 2:
            shape_a = input_x1.get("shape")

    if shape_b is not None:
        if len(shape_b) < 2:
            shape_b = input_x2.get("shape")

    shape_a = list(shape_a)
    shape_b = list(shape_b)

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_a = _get_input_shape(shape_a)
        shape_b = _get_input_shape(shape_b)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)
    util.check_shape_size(shape_a, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_b, SHAPE_SIZE_LIMIT)

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_a = [shape_a[1], shape_a[0]]
        trans_a = bool(1 - trans_a)

    if input_x2.get("format") == "FRACTAL_NZ":
        shape_b = [shape_b[1], shape_b[0]]
        trans_b = bool(1 - trans_b)

    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        shape_bias = _get_bias(shape_bias)

    src_dtype = input_x1.get("dtype").lower()
    dst_dtype = output_y.get("dtype").lower()
    if src_dtype in ("float32", "int32"):
        matmul_vector_cce(shape_a, shape_b, src_dtype, trans_a, trans_b, shape_bias, kernel_name)
        return
    _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b)
    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_a) - 2]
    n_shape = shape_b[len(shape_a) - 1]

    if src_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT

    if trans_a and km_shape == 1:
        block_in = cce.BLOCK_VECTOR

    if not trans_a and m_shape == 1:
        block_in = cce.BLOCK_VECTOR

    if trans_b and kn_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if not trans_b and n_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if trans_a:
        shape_a_temp = (m_shape // block_reduce, km_shape // block_in, block_reduce, block_in)
    else:
        shape_a_temp = (m_shape // block_in, km_shape // block_reduce, block_in, block_reduce)

    if trans_b:
        shape_b_temp = (kn_shape // block_out, n_shape // block_reduce, block_reduce, block_out)
    else:
        shape_b_temp = (kn_shape // block_reduce, n_shape // block_out, block_out, block_reduce)

    if input_x1.get("format") == "FORMAT_FRACTAL_Z":
        shape_a_temp = (shape_a_temp[0], shape_a_temp[1], shape_a_temp[2], shape_a_temp[3])
        format_a = "fractal"
    elif input_x1.get("format") == "FRACTAL_NZ":
        shape_a_temp = (shape_a_temp[0], shape_a_temp[1], shape_a_temp[2], shape_a_temp[3])
        format_a = "FRACTAL_NZ"
    else:
        shape_a_temp = (shape_a[len(shape_a) - 2], shape_a[len(shape_a) - 1])
        format_a = "ND"

    if input_x2.get("format") == "FORMAT_FRACTAL_Z":
        shape_b_temp = (shape_b_temp[0], shape_b_temp[1], shape_b_temp[2], shape_b_temp[3])
        format_b = "fractal"
    elif input_x2.get("format") == "FRACTAL_NZ":
        shape_b_temp = (shape_b_temp[0], shape_b_temp[1], shape_b_temp[2], shape_b_temp[3])
        format_b = "FRACTAL_NZ"
    else:
        shape_b_temp = (shape_b[len(shape_b) - 2], shape_b[len(shape_b) - 1])
        format_b = "ND"

    tensor_bias = None
    tensor_a = tvm.placeholder(shape_a_temp, name='tensor_a',
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b_temp, name='tensor_b',
                               dtype=src_dtype)

    if shape_bias:
        tensor_bias = tvm.placeholder(shape_bias, name='tensor_bias',
                                      dtype=dst_dtype)
    result = te.lang.cce.matmul(tensor_a, tensor_b, trans_a, trans_b, format_a=format_a,
                                format_b=format_b, dst_dtype=dst_dtype, tensor_bias=tensor_bias)

    with tvm.target.cce():
        schedule = generic.auto_schedule(result)

    tensor_list = [tensor_a, tensor_b, result]
    if shape_bias:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)
