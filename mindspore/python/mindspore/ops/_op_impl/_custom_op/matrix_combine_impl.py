# Copyright 2020 Huawei Technologies Co., Ltd
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
"""CusMatrixCombine"""
from __future__ import absolute_import

from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

cus_matrix_combine_op_info = TBERegOp("CusMatrixCombine") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matrixcombine.so") \
    .compute_cost(10) \
    .kernel_name("cus_matrix_combine") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(cus_matrix_combine_op_info)
def cus_matrix_combine(input_x, output, kernel_name="cus_matrix_combine"):
    """CusMatrixCombine"""
    input_x_shape = input_x.get("shape")
    output_shape = output.get("shape")

    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))

    input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
    res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)

    blocks = 32
    matrix_dim = input_x_shape[0] * input_x_shape[1]
    if input_x_shape[0] == 1 and input_x_shape[1] == 64:
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (2, matrix_dim), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[0, block_index * 2, 0], 0, 1, 16, 0, 0)
            tik_instance.data_move(res[block_index * 2, 0], input_x_ub, 0, 1, 16, 0, 0)
    else:
        tiling_dim = 4
        input_x_ub = tik_instance.Tensor("float32", (tiling_dim, matrix_dim), name="input_x_ub",
                                         scope=tik.scope_ubuf)
        zero = tik_instance.Scalar("float32")
        zero.set_as(0.0)
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index, \
                tik_instance.for_range(0, input_x_shape[0]) as i:
            repeat_real = tiling_dim * matrix_dim // 64
            if repeat_real <= 255:
                tik_instance.vector_dup(64, input_x_ub, zero, repeat_real, 1, 8)
            elif repeat_real <= 510:
                tik_instance.vector_dup(64, input_x_ub, zero, 255, 1, 8)
                tik_instance.vector_dup(64, input_x_ub[255 * 64], zero, repeat_real - 255, 1, 8)
            elif repeat_real <= 765:
                tik_instance.vector_dup(64, input_x_ub, zero, 255, 1, 8)
                tik_instance.vector_dup(64, input_x_ub[255 * 64], zero, 255, 1, 8)
                tik_instance.vector_dup(64, input_x_ub[510 * 64], zero, repeat_real - 510, 1, 8)
            else:
                tik_instance.vector_dup(64, input_x_ub, zero, 255, 1, 8)
                tik_instance.vector_dup(64, input_x_ub[255 * 64], zero, 255, 1, 8)
                tik_instance.vector_dup(64, input_x_ub[510 * 64], zero, 255, 1, 8)
                tik_instance.vector_dup(64, input_x_ub[765 * 64], zero, repeat_real - 765, 1, 8)

            with tik_instance.for_range(0, tiling_dim) as j:
                tik_instance.data_move(input_x_ub[j, 128 * i], input_x[i, block_index * tiling_dim + j, 0],
                                       0, 1, 16, 0, 0)
            tik_instance.data_move(res[i * 128 + block_index * tiling_dim, 0], input_x_ub, 0, 1,
                                   tiling_dim * matrix_dim * 4 // 32, 0, 0)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance
