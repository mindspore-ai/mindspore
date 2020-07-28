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
"""CusFusedAbsMax1"""
from te import tik
from topi.cce import util

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

cus_fused_abs_max1_op_info = TBERegOp("CusFusedAbsMax1") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fusedabsmax1.so") \
    .compute_cost(10) \
    .kernel_name("CusFusedAbsMax1") \
    .partial_flag(True) \
    .attr("origin_shape", "required", "listInt", "all") \
    .input(0, "x1", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(cus_fused_abs_max1_op_info)
def CusFusedAbsMax1(input_x, output, origin_shape=None, kernel_name="fused_abs_max1"):
    """CusFusedAbsMax1"""
    input_x_shape = input_x.get("shape")
    output_shape = output.get("shape")
    dtype = input_x.get("dtype")

    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))

    support_shape = [((1, 128, 128), "float32"),
                     ((2, 128, 128), "float32"),
                     ((4, 128, 128), "float32"),
                     ((8, 128, 128), "float32"),
                     ((16, 128, 128), "float32"),
                     ((5, 128, 128), "float32"),
                     ((9, 128, 128), "float32"),
                     ((18, 128, 128), "float32"),
                     ((36, 128, 128), "float32"),
                     ((32, 128, 128), "float32"),
                     ((1, 64, 64), "float32"),
                     ((32, 64), "float32")
                     ]
    ori_shape = tuple(origin_shape)
    input_info = (tuple(input_x_shape), dtype)
    if input_info not in support_shape:
        raise RuntimeError("input_shape %s is not supported" % str(input_info))
    if input_info == ((1, 128, 128), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            repeat_time = each_block_element // 64
            tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((2, 128, 128), "float32"):
        if ori_shape == (147, 147):
            phase_1 = 16384
            phase_2 = 1216
            blocks = 32
            each_block_element = phase_1 // blocks + 64
            input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
            res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
            with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
                input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                                 scope=tik.scope_ubuf)
                broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                           scope=tik.scope_ubuf)
                tik_instance.data_move(input_x_ub, input_x[512 * block_index], 0, 1, 512 // 8, 0, 0)
                line_id = block_index % 19
                tik_instance.data_move(input_x_ub[512], input_x[16384 + 128 * line_id], 0, 1, 8, 0, 0)
                repeat_time = each_block_element // 64
                tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
                tik_instance.vmax(19, input_x_ub, input_x_ub, input_x_ub[512], 1, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
                with tik_instance.for_range(0, 64) as cc0:
                    data_temp = tik_instance.Scalar("float32")
                    data_temp.set_as(input_x_ub[cc0])
                    tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1,
                                  1, 8, 8, 8)
                tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
        elif ori_shape in ((256, 256), None, (-1, -1)):
            input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
            res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
            total_elements = 1
            for val in input_x_shape:
                total_elements *= val
            blocks = 32
            each_block_element = total_elements // blocks
            with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
                input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                                 scope=tik.scope_ubuf)
                broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                           scope=tik.scope_ubuf)
                tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                       each_block_element // 8, 0, 0)
                repeat_time = each_block_element // 64
                tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
                with tik_instance.for_range(0, 64) as cc0:
                    data_temp = tik_instance.Scalar("float32")
                    data_temp.set_as(input_x_ub[cc0])
                    tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1,
                                  1, 8, 8, 8)
                tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
        else:
            raise RuntimeError("origin shape %s is not supported" % str(ori_shape))
    elif input_info == ((4, 128, 128), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            repeat_time = each_block_element // 64
            tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((8, 128, 128), "float32"):
        if ori_shape == (1000, 1000):
            input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
            res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
            blocks = 32
            each_block_element = 7 * 128 * 128 // 32 + 4 * 128
            phase_1 = 7 * 128 * 128 // 32
            with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
                input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                                 scope=tik.scope_ubuf)
                broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                           scope=tik.scope_ubuf)
                tik_instance.data_move(input_x_ub, input_x[phase_1 * block_index], 0, 1, phase_1 // 8, 0, 0)
                tik_instance.data_move(input_x_ub[phase_1], input_x[114688 + block_index * 384], 0, 1, 384 // 8, 0,
                                       0)
                move_idx = block_index % 8
                tik_instance.data_move(input_x_ub[phase_1 + 384], input_x[114688 + 96 * 128 + move_idx * 128], 0, 1,
                                       128 // 8, 0, 0)
                repeat_time = each_block_element // 64
                tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
                vmask = 1000 - 7 * 128 - 64
                with tik_instance.for_range(0, 4) as loop_idx:
                    tik_instance.vmax(vmask, input_x_ub[3584 + 128 * loop_idx], input_x_ub[3584 + 128 * loop_idx],
                                      input_x_ub[3584 + 128 * loop_idx + 64], 1, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub[512], input_x_ub[2048], 24, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)

                with tik_instance.for_range(0, 4) as loop_idx:
                    tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[3584 + 128 * loop_idx], 1, 1, 1, 1, 8,
                                      8, 8)
                with tik_instance.for_range(0, 64) as cc0:
                    data_temp = tik_instance.Scalar("float32")
                    data_temp.set_as(input_x_ub[cc0])
                    tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1,
                                  1, 8, 8, 8)
                tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
        elif ori_shape == (1001, 1001):
            input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
            res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
            blocks = 32
            each_block_element = 7 * 128 * 128 // 32 + 4 * 128
            phase_1 = 7 * 128 * 128 // 32
            with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
                input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                                 scope=tik.scope_ubuf)
                broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                           scope=tik.scope_ubuf)
                tik_instance.data_move(input_x_ub, input_x[phase_1 * block_index], 0, 1, phase_1 // 8, 0, 0)
                tik_instance.data_move(input_x_ub[phase_1], input_x[114688 + block_index * 384], 0, 1, 384 // 8, 0,
                                       0)
                tik_instance.data_move(input_x_ub[phase_1], input_x[114688 + block_index * 384], 0, 1, 384 // 8, 0,
                                       0)
                move_idx = block_index % 9
                tik_instance.data_move(input_x_ub[phase_1 + 384], input_x[114688 + 96 * 128 + move_idx * 128], 0, 1,
                                       128 // 8, 0, 0)
                repeat_time = each_block_element // 64
                tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
                vmask = 1001 - 7 * 128 - 64
                with tik_instance.for_range(0, 4) as loop_idx:
                    tik_instance.vmax(vmask, input_x_ub[3584 + 128 * loop_idx], input_x_ub[3584 + 128 * loop_idx],
                                      input_x_ub[3584 + 128 * loop_idx + 64], 1, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub[512], input_x_ub[2048], 24, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
                with tik_instance.for_range(0, 4) as loop_idx:
                    tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[3584 + 128 * loop_idx], 1, 1, 1, 1, 8,
                                      8, 8)
                with tik_instance.for_range(0, 64) as cc0:
                    data_temp = tik_instance.Scalar("float32")
                    data_temp.set_as(input_x_ub[cc0])
                    tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1,
                                  1, 8, 8, 8)
                tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
        elif ori_shape in ((1024, 1024), None, (-1, -1)):
            input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
            res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
            total_elements = 1
            for val in input_x_shape:
                total_elements *= val
            blocks = 32
            each_block_element = total_elements // blocks
            with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
                input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                                 scope=tik.scope_ubuf)
                broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                           scope=tik.scope_ubuf)
                tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                       each_block_element // 8, 0, 0)
                repeat_time = each_block_element // 64
                tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[2048], 32, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
                with tik_instance.for_range(0, 64) as cc0:
                    data_temp = tik_instance.Scalar("float32")
                    data_temp.set_as(input_x_ub[cc0])
                    tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1,
                                  1, 1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1,
                                  1, 8, 8, 8)
                tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
        else:
            raise RuntimeError("origin shape %s is not supported" % str(ori_shape))
    elif input_info == ((16, 128, 128), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            repeat_time = each_block_element // 64
            tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[4096], 64, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[2048], 32, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((32, 128, 128), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            tik_instance.vabs(64, input_x_ub, input_x_ub, 255, 1, 1, 8, 8)
            tik_instance.vabs(64, input_x_ub[255 * 64], input_x_ub[255 * 64], 1, 1, 1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[8192], 128, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[4096], 64, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[2048], 32, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((5, 128, 128), "float32"):
        if ori_shape == (576, 576):
            input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
            res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
            total_elements = 69632
            blocks = 32
            each_block_element = total_elements // blocks
            phase_1 = 2048
            phase_2 = 128
            with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
                input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                                 scope=tik.scope_ubuf)
                broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                           scope=tik.scope_ubuf)
                tik_instance.data_move(input_x_ub, input_x[phase_1 * block_index], 0, 1, phase_1 // 8, 0, 0)
                tik_instance.data_move(input_x_ub[phase_1], input_x[65536 + phase_2 * block_index * 2], 0, 1, 8, 0, 0)
                tik_instance.data_move(input_x_ub[phase_1 + 64], input_x[65536 + 128 + phase_2 * block_index * 2], 0, 1,
                                       8, 0, 0)
                repeat_time = each_block_element // 64
                tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
                tik_instance.vmax(64, input_x_ub[2048], input_x_ub[2048], input_x_ub[2048 + 64], 1, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
                tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[2048], 1, 1, 1, 1, 8, 8, 8)
                with tik_instance.for_range(0, 64) as cc0:
                    data_temp = tik_instance.Scalar("float32")
                    data_temp.set_as(input_x_ub[cc0])
                    tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                                  1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                                  1, 8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                                  8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                                  8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                                  8, 8, 8)
                tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                                  8, 8, 8)
                tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
        else:
            raise RuntimeError("origin shape %s is not supported" % str(ori_shape))
    elif input_info == ((9, 128, 128), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            repeat_time = each_block_element // 64
            tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[2048], 32, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[4096], input_x_ub[4096], input_x_ub[4096 + 256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[4096], input_x_ub[4096], input_x_ub[4096 + 128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[4096], input_x_ub[4096], input_x_ub[4096 + 64], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[4096], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((18, 128, 128), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            repeat_time = each_block_element // 64
            tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[4096], 64, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[2048], 32, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[8192], input_x_ub[8192], input_x_ub[8192 + 512], 8, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[8192], input_x_ub[8192], input_x_ub[8192 + 256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[8192], input_x_ub[8192], input_x_ub[8192 + 128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[8192], input_x_ub[8192], input_x_ub[8192 + 64], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[8192], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((36, 128, 128), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            repeat_time_1 = 255
            repeat_time_2 = each_block_element // 64 - 255
            tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time_1, 1, 1, 8, 8)
            tik_instance.vabs(64, input_x_ub[repeat_time_1 * 64], input_x_ub[repeat_time_1 * 64], repeat_time_2, 1,
                              1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[8192], 128, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[4096], 64, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[2048], 32, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub[16384], input_x_ub[16384], input_x_ub[16384 + 1024], 16, 1, 1, 1, 8, 8,
                              8)
            tik_instance.vmax(64, input_x_ub[16384], input_x_ub[16384], input_x_ub[16384 + 512], 8, 1, 1, 1, 8, 8,
                              8)
            tik_instance.vmax(64, input_x_ub[16384], input_x_ub[16384], input_x_ub[16384 + 256], 4, 1, 1, 1, 8, 8,
                              8)
            tik_instance.vmax(64, input_x_ub[16384], input_x_ub[16384], input_x_ub[16384 + 128], 2, 1, 1, 1, 8, 8,
                              8)
            tik_instance.vmax(64, input_x_ub[16384], input_x_ub[16384], input_x_ub[16384 + 64], 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[16384], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((1, 64, 64), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        total_elements = 1
        for val in input_x_shape:
            total_elements *= val
        blocks = 32
        each_block_element = total_elements // blocks
        with tik_instance.for_range(0, blocks, block_num=blocks) as block_index:
            input_x_ub = tik_instance.Tensor("float32", (each_block_element,), name="input_x_ub",
                                             scope=tik.scope_ubuf)
            broadcast_0_local_UB = tik_instance.Tensor("float32", (4096,), name="broadcast_0_local_UB",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_x_ub, input_x[each_block_element * block_index], 0, 1,
                                   each_block_element // 8, 0, 0)
            repeat_time = each_block_element // 64
            tik_instance.vabs(64, input_x_ub, input_x_ub, repeat_time, 1, 1, 8, 8)
            tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
            with tik_instance.for_range(0, 64) as cc0:
                data_temp = tik_instance.Scalar("float32")
                data_temp.set_as(input_x_ub[cc0])
                tik_instance.vector_dup(64, broadcast_0_local_UB[cc0 * 64], data_temp, 1, 1, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[2048], 32, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[1024], 16, 1, 1,
                              1, 8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[512], 8, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[256], 4, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[128], 2, 1, 1, 1,
                              8, 8, 8)
            tik_instance.vmax(64, broadcast_0_local_UB, broadcast_0_local_UB, broadcast_0_local_UB[64], 1, 1, 1, 1,
                              8, 8, 8)
            tik_instance.data_move(res[block_index, 0], broadcast_0_local_UB, 0, 1, 8, 0, 0)
    elif input_info == ((32, 64), "float32"):
        input_x = tik_instance.Tensor("float32", input_x_shape, name="input_x", scope=tik.scope_gm)
        res = tik_instance.Tensor("float32", output_shape, name="res", scope=tik.scope_gm)
        input_x_ub = tik_instance.Tensor("float32", (32 * 64,), name="input_x_ub", scope=tik.scope_ubuf)
        tik_instance.data_move(input_x_ub, input_x, 0, 1, 256, 0, 0)
        tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[1024], 16, 1, 1, 1, 8, 8, 8)
        tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[512], 8, 1, 1, 1, 8, 8, 8)
        tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[256], 4, 1, 1, 1, 8, 8, 8)
        tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[128], 2, 1, 1, 1, 8, 8, 8)
        tik_instance.vmax(64, input_x_ub, input_x_ub, input_x_ub[64], 1, 1, 1, 1, 8, 8, 8)
        tik_instance.data_move(res[0], input_x_ub, 0, 1, 1, 0, 0)
    else:
        raise RuntimeError("UnSupportedShape")

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance
