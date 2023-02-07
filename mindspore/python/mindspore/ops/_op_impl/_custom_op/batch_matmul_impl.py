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
"""batch_matmul_impl"""
from __future__ import absolute_import

from te import tik
from tbe.tvm.topi.cce import util

from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

cus_batchmatmul_op_info = TBERegOp("CusBatchMatMul") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("batchmatmul.so") \
    .compute_cost(10) \
    .kernel_name("cus_batch_matmul") \
    .partial_flag(True) \
    .attr("transpose_a", "optional", "bool", "all", "false") \
    .attr("transpose_b", "optional", "bool", "all", "true") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


def _get_flattern_shape(shape):
    """_get_flattern_shape"""
    flattern_shape = 1
    for dim in shape:
        flattern_shape *= dim
    return (flattern_shape,)


def _error_feedback(input_shape):
    """error feedback"""
    support_shape = [((8, 128, 128), (8, 128, 128), "float32", False, True),
                     ((36, 128, 128), (36, 128, 128), "float32", False, True),
                     ((5, 128, 128), (5, 128, 128), "float32", False, True),
                     ((18, 128, 128), (18, 128, 128), "float32", False, True),
                     ((16, 128, 128), (16, 128, 128), "float32", False, True),
                     ((9, 128, 128), (9, 128, 128), "float32", False, True),
                     ((1, 64, 64), (1, 64, 64), "float32", False, True),
                     ((1, 128, 128), (1, 128, 128), "float32", False, True),
                     ((4, 128, 128), (4, 128, 128), "float32", False, True),
                     ((2, 128, 128), (2, 128, 128), "float32", False, True),
                     ((6, 128, 128), (6, 128, 128), "float32", False, True),
                     ((24, 128, 128), (24, 128, 128), "float32", False, True),
                     ((32, 128, 128), (32, 128, 128), 'float32', False, True)]
    if input_shape not in support_shape:
        raise RuntimeError("input_shape %s is not supported" % str(input_shape))


def _inner_matmul_new(tik_instance, dtype, input_info, res, res_index):
    """_inner_matmul_new"""
    input1, input1_index, input2, input2_index = input_info
    input_1_local_ub = tik_instance.Tensor(dtype, [128], name="input_1_local_ub", scope=tik.scope_ubuf)
    t_1_0_local_ub = tik_instance.Tensor(dtype, [64 * 128], name="t_1_0_local_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(input_1_local_ub, input1[input1_index], 0, 1, 16, 0, 0)
    with tik_instance.for_range(0, 2) as vec_i:
        tik_instance.vadds(64, t_1_0_local_ub[vec_i * 64], input_1_local_ub[vec_i * 64], 0, 64, 1, 1, 16, 0)
    with tik_instance.for_range(0, 2, thread_num=2) as thread_idx2:
        input_2_local_ub = tik_instance.Tensor(dtype, [64 * 128], name="input_2_local_ub",
                                               scope=tik.scope_ubuf)
        t_1_local_ub = input_2_local_ub
        bisec_last_axis_local_ub = input_2_local_ub
        matmul_hybrid_f_t_local_ub = tik_instance.Tensor(dtype, [64], name="matmul_hybrid_f_t_local_ub",
                                                         scope=tik.scope_ubuf)
        matmul_hybrid_f_t_local_ub_dst_tmp = tik_instance.Tensor(dtype, [64],
                                                                 name="matmul_hybrid_f_t_local_ub_dst_tmp",
                                                                 scope=tik.scope_ubuf)
        tik_instance.vector_dup(64, matmul_hybrid_f_t_local_ub, 0, 1, 1, 8)
        tik_instance.data_move(input_2_local_ub, input2[input2_index + thread_idx2 * 8192], 0, 1, 1024, 0, 0)
        tik_instance.vmul(64, t_1_local_ub, t_1_0_local_ub, input_2_local_ub, 128, 1, 1, 1, 8, 8, 8)
        tik_instance.vadd(64, bisec_last_axis_local_ub, t_1_local_ub, t_1_local_ub[64], 64, 1, 1, 1,
                          16, 16, 16)
        tik_instance.vector_dup(64, matmul_hybrid_f_t_local_ub_dst_tmp, 0, 1, 1, 8)
        with tik_instance.for_range(0, 64) as cc6:
            tik_instance.vcadd(64, matmul_hybrid_f_t_local_ub_dst_tmp[cc6], bisec_last_axis_local_ub[cc6 * 128],
                               1, 1, 1, 8)
        tik_instance.vadd(64, matmul_hybrid_f_t_local_ub, matmul_hybrid_f_t_local_ub_dst_tmp,
                          matmul_hybrid_f_t_local_ub, 1, 1, 1, 1, 8, 8, 8)
        tik_instance.data_move(res[res_index + thread_idx2 * 64],
                               matmul_hybrid_f_t_local_ub, 0, 1, 8, 0, 0)


def _inner_matmul_new_1_64_32_64(tik_instance, dtype, input_info, res, res_index):
    """_inner_matmul_new_1_64_32_64"""
    input1, input1_index, input2, input2_index = input_info
    input_1_local_ub = tik_instance.Tensor(dtype, [64], name="input_1_local_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(input_1_local_ub, input1[input1_index], 0, 1, 8, 0, 0)
    with tik_instance.for_range(0, 2, thread_num=2) as thread_idx2:
        input_2_local_ub = tik_instance.Tensor(dtype, [32 * 64], name="input_2_local_ub",
                                               scope=tik.scope_ubuf)
        t_1_local_ub = input_2_local_ub
        matmul_hybrid_f_t_local_ub = tik_instance.Tensor(dtype, [32], name="matmul_hybrid_f_t_local_ub",
                                                         scope=tik.scope_ubuf)
        tik_instance.data_move(input_2_local_ub, input2[input2_index + thread_idx2 * 2048], 0, 1, 256, 0, 0)
        tik_instance.vmul(64, t_1_local_ub, input_1_local_ub, input_2_local_ub, 32, 1, 1, 1, 8, 0, 8)
        with tik_instance.for_range(0, 32) as cc6:
            tik_instance.vcadd(64, matmul_hybrid_f_t_local_ub[cc6], t_1_local_ub[cc6 * 64],
                               1, 1, 1, 8)
        tik_instance.data_move(res[res_index + thread_idx2 * 32],
                               matmul_hybrid_f_t_local_ub, 0, 1, 4, 0, 0)


def process_input_shape_640(input_shape, tik_instance, dtype, total_input, res):
    """process input shape of 640"""
    input1, input2 = total_input
    if input_shape == ((5, 128, 128), (5, 128, 128), "float32", False, True):
        with tik_instance.for_range(0, 30, block_num=30) as block_idx, \
                tik_instance.for_range(0, 11) as cc1_db, \
                tik_instance.for_range(0, 2, thread_num=2) as thread_idx, \
                tik_instance.if_scope(((((block_idx % 6) * 22) + (cc1_db * 2) + thread_idx) < 128)):
            input_1_local_ub = tik_instance.Tensor(dtype, [128], name="input_1_local_ub",
                                                   scope=tik.scope_ubuf)
            t_1_0_local_ub = tik_instance.Tensor(dtype, [64 * 128], name="t_1_0_local_ub",
                                                 scope=tik.scope_ubuf)
            tik_instance.data_move(input_1_local_ub, input1[
                (block_idx // 6) * 16384 + (block_idx % 6) * 2816 + cc1_db * 256 + thread_idx * 128], 0, 1,
                                   16, 0, 0)
            with tik_instance.for_range(0, 2) as vec_i:
                tik_instance.vadds(64, t_1_0_local_ub[vec_i * 64], input_1_local_ub[vec_i * 64], 0,
                                   64, 1, 1, 16, 0)
            with tik_instance.for_range(0, 2, thread_num=2) as thread_idx2:
                input_2_local_ub = tik_instance.Tensor(dtype, [64 * 128], name="input_2_local_ub",
                                                       scope=tik.scope_ubuf)
                t_1_local_ub = input_2_local_ub
                bisec_last_axis_local_ub = input_2_local_ub
                matmul_hybrid_f_t_local_ub = tik_instance.Tensor(dtype, [64],
                                                                 name="matmul_hybrid_f_t_local_ub",
                                                                 scope=tik.scope_ubuf)
                matmul_hybrid_f_t_local_ub_dst_tmp = tik_instance.Tensor(
                    dtype, [64],
                    name="matmul_hybrid_f_t_local_ub_dst_tmp",
                    scope=tik.scope_ubuf)
                tik_instance.vector_dup(64, matmul_hybrid_f_t_local_ub, 0, 1, 1, 8)
                tik_instance.data_move(input_2_local_ub,
                                       input2[(block_idx // 6) * 16384 + thread_idx2 * 8192], 0, 1,
                                       1024, 0, 0)
                tik_instance.vmul(64, t_1_local_ub, t_1_0_local_ub,
                                  input_2_local_ub, 128, 1, 1, 1, 8, 8, 8)
                tik_instance.vadd(64, bisec_last_axis_local_ub, t_1_local_ub,
                                  t_1_local_ub[64], 64, 1, 1, 1, 16, 16, 16)
                tik_instance.vector_dup(64, matmul_hybrid_f_t_local_ub_dst_tmp, 0, 1, 1, 8)
                with tik_instance.for_range(0, 64) as cc6:
                    tik_instance.vcadd(64, matmul_hybrid_f_t_local_ub_dst_tmp[cc6],
                                       bisec_last_axis_local_ub[cc6 * 128],
                                       1, 1, 1, 8)
                tik_instance.vadd(64, matmul_hybrid_f_t_local_ub, matmul_hybrid_f_t_local_ub_dst_tmp,
                                  matmul_hybrid_f_t_local_ub, 1, 1, 1, 1, 8, 8, 8)
                tik_instance.data_move(
                    res[(block_idx // 6) * 16384 + (block_idx % 6) * 2816 + cc1_db * 256 +
                        thread_idx * 128 + thread_idx2 * 64],
                    matmul_hybrid_f_t_local_ub, 0, 1, 8, 0, 0)


def process_input_shape_1152(input_shape, tik_instance, dtype, total_input, res):
    """process input shape of 1152"""
    input1, input2 = total_input
    if input_shape == ((9, 128, 128), (9, 128, 128), "float32", False, True):
        with tik_instance.for_range(0, 27, block_num=27) as block_idx:
            with tik_instance.for_range(0, 42, thread_num=2) as cc0:
                input1_index = (block_idx // 3) * 16384 + (block_idx % 3) * 5504 + cc0 * 128
                input2_index = (block_idx // 3) * 16384
                res_index = (block_idx // 3) * 16384 + (block_idx % 3) * 5504 + cc0 * 128
                input_info = input1, input1_index, input2, input2_index
                _inner_matmul_new(tik_instance, dtype, input_info, res, res_index)
            with tik_instance.if_scope((block_idx % 3) < 2):
                input1_index = (block_idx // 3) * 16384 + (block_idx % 3) * 5504 + 42 * 128
                input2_index = (block_idx // 3) * 16384
                res_index = (block_idx // 3) * 16384 + (block_idx % 3) * 5504 + 42 * 128
                input_info = input1, input1_index, input2, input2_index
                _inner_matmul_new(tik_instance, dtype, input_info, res, res_index)


@op_info_register(cus_batchmatmul_op_info)
def cus_batch_matmul(input_x1, input_x2, output, transpose_a=False,
                     transpose_b=True, kernel_name="batchmatmul"):
    """cus_batch_matmul"""
    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
    x1_shape = input_x1.get("shape")
    dtype = input_x1.get("dtype").lower()
    x2_shape = input_x2.get("shape")
    if dtype != input_x2.get("dtype").lower():
        raise RuntimeError("dtype of input_x1 and input_x2 must be same, but got %s vs %s" % (
            dtype, input_x2.get("dtype").lower()))
    input_shape = (tuple(x1_shape), tuple(x2_shape), dtype, transpose_a, transpose_b)

    _error_feedback(input_shape)

    batch, m, k = x1_shape

    input1_shape = _get_flattern_shape(x1_shape)
    input1 = tik_instance.Tensor(dtype, input1_shape, name="input1", scope=tik.scope_gm)
    input2_shape = _get_flattern_shape(x2_shape)
    input2 = tik_instance.Tensor(dtype, input2_shape, name="input2", scope=tik.scope_gm)

    output_shape = x1_shape
    res_shape = _get_flattern_shape(output_shape)
    res = tik_instance.Tensor(dtype, res_shape, name="res", scope=tik.scope_gm)

    if input_shape == ((36, 128, 128), (36, 128, 128), "float32", False, True):
        with tik_instance.for_range(0, 18, block_num=18) as block_idx, \
                tik_instance.for_range(0, 2) as cc0, \
                tik_instance.for_range(0, 128, thread_num=2) as cc1:
            input1_index = block_idx * 32768 + cc0 * 16384 + cc1 * 128
            input2_index = block_idx * 32768 + cc0 * 16384
            res_index = block_idx * 32768 + cc0 * 16384 + cc1 * 128
            input_info = input1, input1_index, input2, input2_index
            _inner_matmul_new(tik_instance, dtype, input_info, res, res_index)

    total_input = input1, input2
    process_input_shape_640(input_shape, tik_instance, dtype, total_input, res)

    if input_shape == ((18, 128, 128), (18, 128, 128), "float32", False, True):
        with tik_instance.for_range(0, 18, block_num=18) as block_idx, \
                tik_instance.for_range(0, 128, thread_num=2) as cc0:
            input1_index = block_idx * 16384 + cc0 * 128
            input2_index = block_idx * 16384
            res_index = block_idx * 16384 + cc0 * 128
            input_info = input1, input1_index, input2, input2_index
            _inner_matmul_new(tik_instance, dtype, input_info, res, res_index)

    process_input_shape_1152(input_shape, tik_instance, dtype, total_input, res)

    if input_shape == ((1, 64, 64), (1, 64, 64), "float32", False, True):
        with tik_instance.for_range(0, 32, block_num=32) as block_idx, \
                tik_instance.for_range(0, 2, thread_num=2) as cc0:
            input1_index = block_idx * 128 + cc0 * 64
            input2_index = 0
            res_index = block_idx * 128 + cc0 * 64
            input_info = input1, input1_index, input2, input2_index
            _inner_matmul_new_1_64_32_64(tik_instance, dtype, input_info,
                                         res, res_index)

    input_shape_list = [((1, 128, 128), (1, 128, 128), "float32", False, True),
                        ((2, 128, 128), (2, 128, 128), "float32", False, True),
                        ((4, 128, 128), (4, 128, 128), "float32", False, True),
                        ((6, 128, 128), (6, 128, 128), "float32", False, True),
                        ((8, 128, 128), (8, 128, 128), "float32", False, True),
                        ((16, 128, 128), (16, 128, 128), "float32", False, True),
                        ((24, 128, 128), (24, 128, 128), "float32", False, True),
                        ((32, 128, 128), (32, 128, 128), 'float32', False, True)]
    if input_shape in input_shape_list:
        block_num, thread_num = 32, 2
        input1_unit_size = 128
        input2_unint_size = 128 * 128
        block_process_ele_num = (batch * m * k) // block_num
        loop_time = (batch * m * k) // block_num // input1_unit_size
        with tik_instance.for_range(0, block_num, block_num=block_num) as block_idx, \
                tik_instance.for_range(0, loop_time, thread_num=thread_num) as cc0:
            input1_index = block_idx * block_process_ele_num + cc0 * input1_unit_size
            if batch > 1:
                input2_index = block_idx // (block_num // batch) * input2_unint_size
            else:
                input2_index = 0
            res_index = block_idx * block_process_ele_num + cc0 * input1_unit_size
            input_info = input1, input1_index, input2, input2_index
            _inner_matmul_new(tik_instance, dtype, input_info,
                              res, res_index)

    tik_instance.BuildCCE(kernel_name, inputs=[input1, input2], outputs=[res])
    return tik_instance
