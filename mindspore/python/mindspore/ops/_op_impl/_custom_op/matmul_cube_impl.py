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
import te.platform.cce_params as cce
import te.lang.cce
from te import tvm
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
from mindspore.ops._op_impl._custom_op._basic import _shape_check, _get_bias, _get_input_shape

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)

matmul_cube_op_info = TBERegOp("CusMatMulCube") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmulcube.so") \
    .compute_cost(10) \
    .kernel_name("cus_matmul_cube") \
    .partial_flag(True) \
    .attr("transpose_a", "required", "bool", "all") \
    .attr("transpose_b", "required", "bool", "all") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_FracNZ, DataType.F16_Default, DataType.F32_FracNZ) \
    .get_op_info()


@op_info_register(matmul_cube_op_info)
def cus_matmul_cube(input_x1, input_x2, bias=None, output_y=None, trans_a=False, trans_b=False,
                    kernel_name="cus_matmul_cube"):
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
    util.check_kernel_name(kernel_name)
    shape_a, shape_b, trans_a, trans_b = _get_shape_a_b(input_x1, input_x2, trans_a, trans_b)

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

    block_in, block_out = _get_block(shape_a, shape_b, trans_a, trans_b)
    shape_a_temp, shape_b_temp, format_a, format_b = _get_format(input_x1, input_x2, trans_a, trans_b,
                                                                 block_in, block_out, shape_a, shape_b)

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

    tensor_list = [tensor_a, tensor_b, result]
    if shape_bias:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    with tvm.target.cce():
        schedule = generic.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(schedule, config)


def _get_shape_a_b(input_x1, input_x2, trans_a, trans_b):
    """_get_shape_a_b"""
    shape_a_x1 = input_x1.get("ori_shape")
    shape_b_x2 = input_x2.get("ori_shape")

    if shape_a_x1 and len(shape_a_x1) < 2:
        shape_a_x1 = input_x1.get("shape")

    if shape_b_x2 and len(shape_b_x2) < 2:
        shape_b_x2 = input_x2.get("shape")

    shape_a_list = list(shape_a_x1)
    shape_b_list = list(shape_b_x2)

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_a = _get_input_shape(shape_a_list)
        shape_b = _get_input_shape(shape_b_list)

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
    return shape_a, shape_b, trans_a, trans_b


def _get_block(shape_a, shape_b, trans_a, trans_b):
    """_get_block"""
    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_a) - 2]
    n_shape = shape_b[len(shape_a) - 1]
    block_out = cce.BLOCK_OUT
    block_in = cce.BLOCK_IN

    if trans_b and kn_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if not trans_b and n_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if trans_a and km_shape == 1:
        block_in = cce.BLOCK_VECTOR

    if not trans_a and m_shape == 1:
        block_in = cce.BLOCK_VECTOR

    return block_in, block_out


def _get_format(input_x1, input_x2, trans_a, trans_b, block_in, block_out, shape_a, shape_b):
    """_get_block"""
    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_a) - 2]
    n_shape = shape_b[len(shape_a) - 1]
    if input_x1.get("dtype").lower() == "float16":
        block_reduce = cce.BLOCK_REDUCE
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

    return shape_a_temp, shape_b_temp, format_a, format_b
