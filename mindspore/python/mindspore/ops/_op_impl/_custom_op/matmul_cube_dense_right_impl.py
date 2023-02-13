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

from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

matmul_cube_dense_right_op_info = TBERegOp("CusMatMulCubeDenseRight") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmulcubedenseright.so") \
    .compute_cost(10) \
    .kernel_name("cus_matmul_cube_dense_right") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_FracNZ, DataType.F16_Default, DataType.F32_Default,
                  DataType.F32_FracNZ) \
    .get_op_info()


@op_info_register(matmul_cube_dense_right_op_info)
def cus_matmul_cube_dense_right(input_x1, input_x2, input_x3, output_y=None,
                                kernel_name="cus_matmul_cube_dense_right"):
    """CusMatMulCubeDenseRight"""
    shape_a_temp = (128, 63, 16, 16)
    shape_b_temp = (128, 128, 16, 16)
    shape_output = output_y.get("shape")
    matrix_max_shape = (1,)
    support_shape = [(shape_a_temp, shape_b_temp, matrix_max_shape),]
    shape_a_input = input_x1.get("shape")
    shape_b_input = input_x2.get("shape")
    matrix_max_input = input_x3.get("shape")
    input_shape = (tuple(shape_a_input), tuple(shape_b_input), tuple(matrix_max_input))
    if input_shape not in support_shape:
        raise RuntimeError("input_shape %s is not supported" % str(input_shape))

    if shape_a_temp[0] == 128 and shape_a_temp[1] == 63 and shape_b_temp[0] == 128 and shape_b_temp[1] == 128:
        if util.get_product_version() == util.VERSION_MINI:
            tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        else:
            tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
        input_x1 = tik_instance.Tensor("float16", shape_a_temp, name="left_matrix", scope=tik.scope_gm)
        input_x2 = tik_instance.Tensor("float16", shape_b_temp, name="right_matrix", scope=tik.scope_gm)
        input_x3 = tik_instance.Tensor("float32", [1,], name="matrix_max", scope=tik.scope_gm)
        resmatmul = tik_instance.Tensor("float32", shape_output, name="output", scope=tik.scope_gm)
        with tik_instance.for_range(0, 32, block_num=32) as block_index:
            core_m_idx = block_index // 16
            core_n_idx = block_index % 16
            matrix_max_scalar = tik_instance.Scalar("float32")
            matrix_max_local_ub = tik_instance.Tensor("float32", (8,), scope=tik.scope_ubuf,
                                                      name="matrix_max_local_ub")
            tik_instance.data_move(matrix_max_local_ub, input_x3, 0, 1, 1, 0, 0)
            matrix_max_scalar.set_as(matrix_max_local_ub[0])

            with tik_instance.if_scope(core_m_idx == 0):
                tik_instance, resmatmul = _update_tik1(tik_instance, input_x1, input_x2, core_n_idx,
                                                       resmatmul, matrix_max_scalar)
            with tik_instance.else_scope():
                tik_instance, resmatmul = _update_tik2(tik_instance, input_x1, input_x2, core_n_idx,
                                                       resmatmul, matrix_max_scalar)

        tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x1, input_x2, input_x3], outputs=[resmatmul])
        return tik_instance
    return None


def _update_tik1(tik_instance, input_x1, input_x2, core_n_idx, resmatmul, matrix_max_scalar):
    """_update_tik1"""
    resmatmul_local_ub = tik_instance.Tensor("float32", (256 * 128,), scope=tik.scope_ubuf,
                                             name="resmatmul_local_ub")
    resmatmul_local_ub_local_l0c = tik_instance.Tensor("float32", (256 * 128,), scope=tik.scope_cc,
                                                       name="resmatmul_local_ub_local_l0c")
    input_1_local_l1_local_l0a = tik_instance.Tensor("float16", (256 * 128,), scope=tik.scope_ca,
                                                     name="input_1_local_l1_local_l0a")
    input_2_local_l1 = tik_instance.Tensor("float16", (8 * 128 * 16,), scope=tik.scope_cbuf,
                                           name="input_2_local_l1")

    input_1_local_l1 = tik_instance.Tensor("float16", (8 * 256 * 16,), scope=tik.scope_cbuf,
                                           name="input_1_local_l1")
    input_2_local_l1_local_l0b = tik_instance.Tensor("float16", (128 * 128,), scope=tik.scope_cb,
                                                     name="input_2_local_l1_local_l0b")
    with tik_instance.for_range(0, 2) as cc1:
        tik_instance.data_move(input_2_local_l1, input_x2[core_n_idx * 262144 + core_n_idx * 2048], 0, 8,
                               128, 1920, 0)
        tik_instance.data_move(input_1_local_l1, input_x1[core_n_idx * 129024 + cc1 * 4096], 0, 8, 256,
                               752, 0)
        with tik_instance.for_range(0, 8) as cc10:
            tik_instance.load2dv1(input_2_local_l1_local_l0b[cc10 * 2048], input_2_local_l1[cc10 * 256], 0,
                                  8, 8, 0, True)
        with tik_instance.for_range(0, 16) as cc101:
            tik_instance.load2dv1(input_1_local_l1_local_l0a[cc101 * 2048], input_1_local_l1[cc101 * 256],
                                  0, 8, 16, 0, False)

        tik_instance.mmad(resmatmul_local_ub_local_l0c, input_1_local_l1_local_l0a,
                          input_2_local_l1_local_l0b, 256, 128, 128, 0)
        tik_instance.data_move(resmatmul_local_ub, resmatmul_local_ub_local_l0c, 0, 1, 128, 0, 0)
        tik_instance.vmuls(64, resmatmul_local_ub, resmatmul_local_ub, matrix_max_scalar, 255, 1, 1, 8, 8)
        tik_instance.vmuls(64, resmatmul_local_ub[255 * 64], resmatmul_local_ub[255 * 64],
                           matrix_max_scalar, 255, 1, 1, 8, 8)
        tik_instance.vmuls(64, resmatmul_local_ub[510 * 64], resmatmul_local_ub[510 * 64],
                           matrix_max_scalar, 2, 1, 1, 8, 8)

        tik_instance.data_move(resmatmul[core_n_idx * 129024 + cc1 * 4096], resmatmul_local_ub, 0, 8, 512,
                               0, 1504)
    return tik_instance, resmatmul


def _update_tik2(tik_instance, input_x1, input_x2, core_n_idx, resmatmul, matrix_max_scalar):
    """_update_tik2"""
    resmatmul_local_ub = tik_instance.Tensor("float32", (256 * 128,), scope=tik.scope_ubuf,
                                             name="resmatmul_local_ub")
    resmatmul_local_ub1 = tik_instance.Tensor("float32", (240 * 128,), scope=tik.scope_ubuf,
                                              name="resmatmul_local_ub1")

    resmatmul_local_ub_local_l0c = tik_instance.Tensor("float32", (256 * 128,), scope=tik.scope_cc,
                                                       name="resmatmul_local_ub_local_l0c")
    resmatmul_local_ub_local_l0c1 = tik_instance.Tensor("float32", (240 * 128,), scope=tik.scope_cc,
                                                        name="resmatmul_local_ub_local_l0c1")

    input_1_local_l1_local_l0a = tik_instance.Tensor("float16", (256 * 128,), scope=tik.scope_ca,
                                                     name="input_1_local_l1_local_l0a")
    input_2_local_l1 = tik_instance.Tensor("float16", (8 * 128 * 16,), scope=tik.scope_cbuf,
                                           name="input_2_local_l1")
    input_2_local_l11 = tik_instance.Tensor("float16", (8 * 128 * 16,), scope=tik.scope_cbuf,
                                            name="input_2_local_l11")

    input_1_local_l1 = tik_instance.Tensor("float16", (8 * 256 * 16,), scope=tik.scope_cbuf,
                                           name="input_1_local_l1")
    input_1_local_l11 = tik_instance.Tensor("float16", (8 * 240 * 16,), scope=tik.scope_cbuf,
                                            name="input_1_local_l11")

    input_2_local_l1_local_l0b = tik_instance.Tensor("float16", (128 * 128,), scope=tik.scope_cb,
                                                     name="input_2_local_l1_local_l0b")
    input_2_local_l1_local_l0b1 = tik_instance.Tensor("float16", (128 * 128,), scope=tik.scope_cb,
                                                      name="input_2_local_l1_local_l0b1")

    tik_instance.data_move(input_2_local_l1, input_x2[core_n_idx * 262144 + core_n_idx * 2048], 0, 8, 128,
                           1920, 0)
    tik_instance.data_move(input_1_local_l1, input_x1[core_n_idx * 129024 + 2 * 4096], 0, 8, 256, 752, 0)
    with tik_instance.for_range(0, 8) as cc10:
        tik_instance.load2dv1(input_2_local_l1_local_l0b[cc10 * 2048], input_2_local_l1[cc10 * 256], 0, 8,
                              8, 0, True)
    with tik_instance.for_range(0, 16) as cc101:
        tik_instance.load2dv1(input_1_local_l1_local_l0a[cc101 * 2048], input_1_local_l1[cc101 * 256],
                              0, 8, 16, 0, False)

    tik_instance.mmad(resmatmul_local_ub_local_l0c, input_1_local_l1_local_l0a, input_2_local_l1_local_l0b,
                      256, 128, 128, 0)
    tik_instance.data_move(resmatmul_local_ub, resmatmul_local_ub_local_l0c, 0, 1, 128, 0, 0)
    tik_instance.vmuls(64, resmatmul_local_ub, resmatmul_local_ub, matrix_max_scalar, 255, 1, 1, 8, 8)
    tik_instance.vmuls(64, resmatmul_local_ub[255 * 64], resmatmul_local_ub[255 * 64], matrix_max_scalar,
                       255, 1, 1, 8, 8)
    tik_instance.vmuls(64, resmatmul_local_ub[510 * 64], resmatmul_local_ub[510 * 64], matrix_max_scalar,
                       2, 1, 1, 8, 8)

    tik_instance.data_move(resmatmul[core_n_idx * 129024 + 2 * 4096], resmatmul_local_ub, 0, 8, 512, 0,
                           1504)

    tik_instance.data_move(input_2_local_l11, input_x2[core_n_idx * 262144 + core_n_idx * 2048], 0, 8, 128,
                           1920, 0)
    tik_instance.data_move(input_1_local_l11, input_x1[core_n_idx * 129024 + 12288], 0, 8, 240, 768, 0)

    with tik_instance.for_range(0, 8) as cc102:
        tik_instance.load2dv1(input_2_local_l1_local_l0b1[cc102 * 2048], input_2_local_l11[cc102 * 256], 0,
                              8, 8, 0, True)
    with tik_instance.for_range(0, 16) as cc103:
        tik_instance.load2dv1(input_1_local_l1_local_l0a[cc103 * 2048], input_1_local_l11[cc103 * 256], 0,
                              8, 15, 0, False)

    tik_instance.mmad(resmatmul_local_ub_local_l0c1, input_1_local_l1_local_l0a,
                      input_2_local_l1_local_l0b1, 240, 128, 128, 0)
    tik_instance.data_move(resmatmul_local_ub1, resmatmul_local_ub_local_l0c1, 0, 1, 120, 0, 0)

    tik_instance.vmuls(64, resmatmul_local_ub1, resmatmul_local_ub1, matrix_max_scalar, 255, 1, 1, 8, 8)
    tik_instance.vmuls(64, resmatmul_local_ub1[255 * 64], resmatmul_local_ub1[255 * 64], matrix_max_scalar,
                       225, 1, 1, 8, 8)

    tik_instance.data_move(resmatmul[core_n_idx * 129024 + 12288], resmatmul_local_ub1, 0, 8, 480, 0, 1536)
    return tik_instance, resmatmul
