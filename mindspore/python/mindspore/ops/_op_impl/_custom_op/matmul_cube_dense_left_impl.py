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
import te.lang.cce
import te.platform.cce_params as cce
from te import tik
from te import tvm
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
from mindspore.ops._op_impl._custom_op._basic import _shape_check, _get_bias, _get_input_shape

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)

matmul_cube_dense_left_op_info = TBERegOp("CusMatMulCubeDenseLeft") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmulcubedenseleft.so") \
    .compute_cost(10) \
    .kernel_name("cus_matmul_cube_dense_left") \
    .partial_flag(True) \
    .attr("trans_a", "optional", "bool", "all", "false") \
    .attr("trans_b", "optional", "bool", "all", "false") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_FracNZ, DataType.F16_Default, DataType.F16_FracNZ) \
    .get_op_info()


def shape_gen1(input_x1, input_x2, output_y, kernel_name, trans_a, trans_b):
    """shape gen1"""
    shape_a_x1 = input_x1.get("ori_shape")
    shape_b_x2 = input_x2.get("ori_shape")
    shape_output = output_y.get("ori_shape")

    if input_x2.get("format") == "FRACTAL_Z":
        n, c, h, w = shape_b_x2
        c0 = 16
        c1 = c // c0
        if c1 == 0:
            c1 = 1
        shape_b_x2 = [n, c1 * h * w * c0]
        shape_a_x1 = [n, n]

    if input_x1.get("format") == "FRACTAL_Z":
        n, c, h, w = shape_a_x1
        c0 = 16
        c1 = c // c0
        if c1 == 0:
            c1 = 1
        shape_a_x1 = [n, c1 * h * w * c0]
        shape_b_x2 = [c1 * h * w * c0, c1 * h * w * c0]

    if input_x2.get("format") == "FRACTAL_NZ":
        shape_a_x1 = [shape_b_x2[0], shape_b_x2[0]]

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_b_x2 = [shape_a_x1[1], shape_a_x1[1]]

    shape_a_list = list(shape_a_x1)
    shape_b_list = list(shape_b_x2)

    shape_a = _get_input_shape(shape_a_list)
    shape_b = _get_input_shape(shape_b_list)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)
    util.check_shape_size(shape_a, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_b, SHAPE_SIZE_LIMIT)

    shape_a = [shape_a[1], shape_a[0]]
    trans_a = bool(1 - trans_a)

    shape_b = [shape_b[1], shape_b[0]]
    trans_b = bool(1 - trans_b)
    return shape_a, shape_b, trans_a, trans_b, shape_output


def shape_gen2(bias, input_x1, output_y, shape_a, shape_b, trans_a, trans_b):
    """shape gen2"""
    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        shape_bias = _get_bias(shape_bias)

    src_dtype = input_x1.get("dtype").lower()
    dst_dtype = output_y.get("dtype").lower()
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
    shape_a_temp = (shape_a_temp[0], shape_a_temp[1], shape_a_temp[2], shape_a_temp[3])
    format_a = "FRACTAL_NZ"
    shape_b_temp = (shape_b_temp[0], shape_b_temp[1], shape_b_temp[2], shape_b_temp[3])
    format_b = "FRACTAL_NZ"
    return shape_a_temp, format_a, shape_b_temp, format_b, shape_bias, src_dtype, dst_dtype


def core(shape_a_temp, shape_b_temp, shape_output, kernel_name):
    """core func"""
    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))

    input_x1 = tik_instance.Tensor("float16", shape_a_temp, name="left_matrix", scope=tik.scope_gm)
    input_x2 = tik_instance.Tensor("float16", shape_b_temp, name="right_matrix", scope=tik.scope_gm)
    resmatmul = tik_instance.Tensor("float16", shape_output, name="output", scope=tik.scope_gm)
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        resmatmul_local_ub = tik_instance.Tensor("float16", (128 * 256,), scope=tik.scope_ubuf,
                                                 name="resmatmul_local_ub")
        resmatmul_local_ub_local_l0c = tik_instance.Tensor("float32", (128 * 256,), scope=tik.scope_cc,
                                                           name="resmatmul_local_ub")
        input_1_local_l1_local_l0a = tik_instance.Tensor("float16", (128 * 128,), scope=tik.scope_ca,
                                                         name="input_1_local_l1_local_l0a")
        input_2_local_l1 = tik_instance.Tensor("float16", (128 * 256,), scope=tik.scope_cbuf,
                                               name="input_2_local_l1")
        input_1_local_l1 = tik_instance.Tensor("float16", (128 * 128,), scope=tik.scope_cbuf,
                                               name="input_1_local_l1")
        input_2_local_l1_local_l0b = tik_instance.Tensor("float16", (128 * 256,), scope=tik.scope_cb,
                                                         name="input_2_local_l1_local_l0b")
        core_m_idx = block_index % 8
        core_n_idx = block_index // 8
        with tik_instance.if_scope(core_m_idx != 7):
            tik_instance.data_move(input_1_local_l1, input_x1[core_m_idx * (8 * 256 + 128 * 1008)], 0, 8, 128,
                                   55 * 16, 0)
            tik_instance.data_move(input_2_local_l1, input_x2[core_m_idx * 8 * 256 + core_n_idx * 512 * 1008], 0,
                                   32, 128, 55 * 16, 0)
            with tik_instance.for_range(0, 8) as cc12:
                tik_instance.load2dv1(input_1_local_l1_local_l0a[cc12 * 2048], input_1_local_l1[cc12 * 256], 0, 8,
                                      8, 0, False)
            with tik_instance.for_range(0, 2) as cc6:
                with tik_instance.for_range(0, 8) as cc121:
                    tik_instance.load2dv1(input_2_local_l1_local_l0b[cc121 * 4096],
                                          input_2_local_l1[cc6 * 32768 + cc121 * 256], 0, 16, 8, 0, True)
                tik_instance.mmad(resmatmul_local_ub_local_l0c, input_1_local_l1_local_l0a,
                                  input_2_local_l1_local_l0b, 128, 128, 256, 0)
                tik_instance.data_move(resmatmul_local_ub, resmatmul_local_ub_local_l0c, 0, 1, 128, 0, 0, 1)
                tik_instance.data_move(resmatmul[cc6 * 256 * 1008 + core_m_idx * 8 * 256 + core_n_idx * 512 * 1008]
                                       , resmatmul_local_ub, 0, 16, 256 // 2, 0, 55 * 16 * 2 // 2)
        with tik_instance.else_scope():
            tik_instance.data_move(input_1_local_l1, input_x1[core_m_idx * (8 * 256 + 128 * 1008)], 0, 7, 112,
                                   56 * 16, 0)
            tik_instance.data_move(input_2_local_l1, input_x2[core_m_idx * 8 * 256 + core_n_idx * 512 * 1008], 0,
                                   32, 112, 56 * 16, 0)
            with tik_instance.for_range(0, 7) as cc10:
                tik_instance.load2dv1(input_1_local_l1_local_l0a[cc10 * 1792], input_1_local_l1[cc10 * 256], 0, 7,
                                      7, 0, False)
            with tik_instance.for_range(0, 2) as cc5:
                with tik_instance.for_range(0, 7) as cc101:
                    tik_instance.load2dv1(input_2_local_l1_local_l0b[cc101 * 4096],
                                          input_2_local_l1[cc5 * 28672 + cc101 * 256], 0, 16, 7, 0, True)
                tik_instance.mmad(resmatmul_local_ub_local_l0c, input_1_local_l1_local_l0a,
                                  input_2_local_l1_local_l0b, 112, 112, 256, 0)
                tik_instance.data_move(resmatmul_local_ub, resmatmul_local_ub_local_l0c, 0, 1, 112, 0, 0, 1)
                tik_instance.data_move(resmatmul[cc5 * 256 * 1008 + core_m_idx * 8 * 256 + core_n_idx * 512 * 1008]
                                       , resmatmul_local_ub, 0, 16, 224 // 2, 0, 56 * 16 * 2 // 2)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x1, input_x2], outputs=[resmatmul])
    return tik_instance


@op_info_register(matmul_cube_dense_left_op_info)
def cus_matmul_cube_dense_left(input_x1, input_x2, bias=None, output_y=None, trans_a=False, trans_b=False,
                               kernel_name="cus_matmul_cube_dense_left"):
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
    shape_a, shape_b, trans_a, trans_b, shape_output = shape_gen1(input_x1, input_x2, output_y, kernel_name,
                                                                  trans_a, trans_b)
    shape_a_temp, format_a, shape_b_temp, format_b, shape_bias, src_dtype, dst_dtype = shape_gen2(bias, input_x1,
                                                                                                  output_y, shape_a,
                                                                                                  shape_b, trans_a,
                                                                                                  trans_b)
    tensor_bias = None
    tensor_a = tvm.placeholder(shape_a_temp, name='tensor_a',
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b_temp, name='tensor_b',
                               dtype=src_dtype)

    if shape_bias:
        tensor_bias = tvm.placeholder(shape_bias, name='tensor_bias',
                                      dtype=dst_dtype)

    if shape_a_temp[0] == 63 and shape_a_temp[1] == 63 and shape_b_temp[0] == 128 and shape_b_temp[1] == 63:
        tik_instance = core(shape_a_temp, shape_b_temp, shape_output, kernel_name)
        return tik_instance

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
    return None
