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
"""CusCholeskyTrsm"""
from __future__ import absolute_import
import logging

from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

cus_cholesky_trsm_op_info = TBERegOp("CusCholeskyTrsm") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("choleskytrsm.so") \
    .compute_cost(10) \
    .kernel_name("cus_cholesky_trsm") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(cus_cholesky_trsm_op_info)
def cus_cholesky_trsm(input_x, output, kernel_name):
    """cus_cholesky_trsm"""
    input_x_shape = input_x.get("shape")
    output_shape = output.get("shape")
    split_dim = 128
    matrix_dim = input_x_shape[0]
    split_dim = min(matrix_dim, split_dim)
    vector_repeat_times = int(split_dim // 64)
    blocks = int(matrix_dim // split_dim)
    if blocks == 0:
        blocks = 1
    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))

    input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
    res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
    with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
        input_x_ub = tik_instance.Tensor("float32", (split_dim, split_dim), name="input_x_ub", scope=tik.scope_ubuf)
        temp_ub = tik_instance.Tensor("float32", (split_dim, split_dim), name="temp_ub", scope=tik.scope_ubuf)
        assist_1_ub = tik_instance.Tensor("float32", (split_dim,), name="assist_1_ub", scope=tik.scope_ubuf)
        assist_2_ub = tik_instance.Tensor("float32", (split_dim,), name="assist_2_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, split_dim) as i:
            tik_instance.data_move(input_x_ub[i, 0], input_x[block_index * split_dim + i, block_index * split_dim], 0,
                                   1, vector_repeat_times * 8, 0, 0)
        scalar1 = tik_instance.Scalar("float32", init_value=-0.5)

        with tik_instance.for_range(0, split_dim) as i:
            scalar2 = tik_instance.Scalar("float32")
            tik_instance.vln(64, assist_1_ub[0], input_x_ub[i, 0], vector_repeat_times, 1, 1, 8, 8)
            tik_instance.vmuls(64, assist_2_ub[0], assist_1_ub[0], scalar1, vector_repeat_times, 1, 1, 8, 8)
            tik_instance.vexp(64, assist_1_ub[0], assist_2_ub[0], vector_repeat_times, 1, 1, 8, 8)
            scalar2.set_as(assist_1_ub[i])
            tik_instance.vmuls(64, input_x_ub[i, 0], input_x_ub[i, 0], scalar2, vector_repeat_times, 1, 1, 8, 8)
            with tik_instance.for_range(i + 1, split_dim) as j:
                scalar3 = tik_instance.Scalar("float32")
                scalar3.set_as(input_x_ub[i, j])
                tik_instance.vmuls(64, temp_ub[j, 0], input_x_ub[i, 0], scalar3, vector_repeat_times, 1, 1, 8, 8)
            tik_instance.vsub(64, input_x_ub[i + 1, 0], input_x_ub[i + 1, 0], temp_ub[i + 1, 0],
                              (split_dim - 1 - i) * vector_repeat_times, 1, 1, 1, 8, 8, 8)

        zero = tik_instance.Scalar("float32")
        zero.set_as(0.0)
        one = tik_instance.Scalar("float32")
        one.set_as(1.0)
        with tik_instance.for_range(0, split_dim) as i:
            tik_instance.vector_dup(64, temp_ub[i, 0], zero, vector_repeat_times, 1, 8)
            temp_ub.__setitem__(i * split_dim + i, one)

        chol_diag_element_final = tik_instance.Scalar("float32")
        chol_diag_element_final.set_as(input_x_ub[split_dim * split_dim - 1])
        trsm_diag_element = tik_instance.Scalar("float32")
        trsm_diag_element.set_as(1.0 / chol_diag_element_final)
        temp_ub.__setitem__(split_dim * split_dim - 1, trsm_diag_element)

        with tik_instance.for_range(1, split_dim) as i:
            index = split_dim - i - 1
            tik_instance.vector_dup(64, assist_1_ub, zero, vector_repeat_times, 1, 8)
            with tik_instance.for_range(0, i) as j:
                chol_diag_element_loop = tik_instance.Scalar("float32")
                chol_diag_element_loop.set_as(input_x_ub[index, index + 1 + j])
                tik_instance.vmuls(64, assist_2_ub, temp_ub[j + index + 1, 0], chol_diag_element_loop,
                                   vector_repeat_times, 1, 1, 8, 8)
                tik_instance.vadd(64, assist_1_ub, assist_2_ub, assist_1_ub, vector_repeat_times, 1, 1, 1, 8, 8, 8)
            temp_scalar = tik_instance.Scalar("float32")
            temp_scalar.set_as(input_x_ub[index, index])
            chol_diag_element = tik_instance.Scalar("float32")
            chol_diag_element.set_as(1.0 / temp_scalar)
            tik_instance.vsub(64, temp_ub[index, 0], temp_ub[index, 0], assist_1_ub, vector_repeat_times,
                              1, 1, 1, 8, 8, 8)
            tik_instance.vmuls(64, temp_ub[index, 0], temp_ub[index, 0], chol_diag_element, vector_repeat_times, 1, 1,
                               8, 8)

        tik_instance.data_move(res[block_index, 0, 0], temp_ub, 0, 1, 8 * vector_repeat_times * split_dim, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance
