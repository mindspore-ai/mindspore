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
"""CusTranspose02314"""
from __future__ import absolute_import

from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

cus_transpose02314_op_info = TBERegOp("CusTranspose02314") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("transpose02314.so") \
    .compute_cost(10) \
    .kernel_name("cus_transpose02314") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_Default) \
    .get_op_info()


def _get_tik():
    """_get_tik"""
    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
    return tik_instance


def _error_feedback(input_x_shape):
    """error feedback"""
    support_shape = [(32, 128, 7, 7, 16),
                     (32, 32, 7, 7, 16),
                     (32, 32, 14, 14, 16),
                     (32, 64, 14, 14, 16),
                     (32, 16, 14, 14, 16),
                     (32, 16, 28, 28, 16),
                     (32, 32, 28, 28, 16),
                     (32, 8, 28, 28, 16),
                     (32, 8, 56, 56, 16),
                     (32, 16, 56, 56, 16),
                     (32, 4, 56, 56, 16),
                     (32, 4, 112, 112, 16)]
    if input_x_shape not in support_shape:
        raise RuntimeError("input_shape %s is not supported" % str(input_x_shape))


@op_info_register(cus_transpose02314_op_info)
def cus_transpose02314(input_x, output, kernel_name="cus_transpose021354"):
    """CusTranspose02314"""
    input_x_shape = input_x.get("shape")
    output_shape = output.get("shape")
    input_x_shape = tuple(input_x_shape)

    _error_feedback(input_x_shape)

    tik_instance = _get_tik()

    input_x = tik_instance.Tensor("float16", input_x_shape, name="input_x", scope=tik.scope_gm)
    res = tik_instance.Tensor("float16", output_shape, name="res", scope=tik.scope_gm)

    dtype = "float16"
    if tuple(input_x_shape) == (32, 4, 112, 112, 16):
        tik_instance, res = shape0(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 4, 56, 56, 16):
        tik_instance, res = shape1(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 16, 56, 56, 16):
        tik_instance, res = shape2(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 8, 56, 56, 16):
        tik_instance, res = shape3(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 8, 28, 28, 16):
        tik_instance, res = shape4(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 32, 28, 28, 16):
        tik_instance, res = shape5(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 16, 28, 28, 16):
        tik_instance, res = shape6(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 16, 14, 14, 16):
        tik_instance, res = shape7(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 128, 7, 7, 16):
        tik_instance, res = shape8(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 32, 7, 7, 16):
        tik_instance, res = shape9(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 32, 14, 14, 16):
        tik_instance, res = shape10(tik_instance, input_x, res, dtype)
    elif tuple(input_x_shape) == (32, 64, 14, 14, 16):
        tik_instance, res = shape11(tik_instance, input_x, res, dtype)

    tik_instance.BuildCCE(kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance


def shape0(tik_instance, input_x, res, dtype):
    """input shape (32, 4, 112, 112, 16)"""
    with tik_instance.for_range(0, 32, block_num=32) as block_idx, tik_instance.for_range(0, 14) as cc1_db, \
            tik_instance.for_range(0, 2, thread_num=2) as db_idx:
        input_1_local_ub = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub",
                                               scope=tik.scope_ubuf)
        t_transpose_local_ub = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub",
                                                   scope=tik.scope_ubuf)
        zero = tik_instance.Scalar(dtype="float16", init_value=0)
        tik_instance.data_move(input_1_local_ub,
                               input_x[block_idx * 802816 + cc1_db * 14336 + 7168 * db_idx], 0, 4, 448,
                               12096, 0)
        with tik_instance.for_range(0, 448) as cc7, tik_instance.for_range(0, 4) as cc8:
            tik_instance.vadds(16, t_transpose_local_ub[cc7 * 64 + cc8 * 16],
                               input_1_local_ub[7168 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 802816 + cc1_db * 57344 + 28672 * db_idx],
                               t_transpose_local_ub, 0, 1, 1792, 0, 0)
    return tik_instance, res


def shape1(tik_instance, input_x, res, dtype):
    """input shape (32, 4, 56, 56, 16)"""
    with tik_instance.for_range(0, 32, block_num=32) as block_idx:
        zero = tik_instance.Scalar(dtype="float16", init_value=0)
        with tik_instance.for_range(0, 3) as cc1_db, tik_instance.for_range(0, 2, thread_num=2) as db_idx:
            input_1_local_ub = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub",
                                                   scope=tik.scope_ubuf)
            t_transpose_local_ub = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_1_local_ub,
                                   input_x[block_idx * 200704 + cc1_db * 14336 + 7168 * db_idx], 0, 4, 448,
                                   2688, 0)
            with tik_instance.for_range(0, 448) as cc7, tik_instance.for_range(0, 4) as cc8:
                tik_instance.vadds(16, t_transpose_local_ub[cc7 * 64 + cc8 * 16],
                                   input_1_local_ub[7168 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
            tik_instance.data_move(res[block_idx * 200704 + cc1_db * 57344 + 28672 * db_idx],
                                   t_transpose_local_ub, 0, 1, 1792, 0, 0)

        input_1_local_ub2 = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub2", scope=tik.scope_ubuf)
        t_transpose_local_ub2 = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub2",
                                                    scope=tik.scope_ubuf)
        tik_instance.data_move(input_1_local_ub2, input_x[block_idx * 200704 + 43008], 0, 4, 448, 2688, 0)
        with tik_instance.for_range(0, 448) as cc72, tik_instance.for_range(0, 4) as cc82:
            tik_instance.vadds(16, t_transpose_local_ub2[cc72 * 64 + cc82 * 16],
                               input_1_local_ub2[7168 * cc82 + cc72 * 16], zero, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 200704 + 172032], t_transpose_local_ub2, 0, 1, 1792, 0, 0)
    return tik_instance, res


def shape2(tik_instance, input_x, res, dtype):
    """input shape (32, 16, 56, 56, 16)"""
    zero = tik_instance.Scalar(dtype="float16", init_value=0)
    with tik_instance.for_range(0, 32, block_num=32) as block_idx, tik_instance.for_range(0, 14) as cc1_db, \
            tik_instance.for_range(0, 2, thread_num=2) as db_idx:
        input_1_local_ub = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub",
                                               scope=tik.scope_ubuf)
        t_transpose_local_ub = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub",
                                                   scope=tik.scope_ubuf)
        tik_instance.data_move(input_1_local_ub,
                               input_x[block_idx * 802816 + cc1_db * 3584 + 1792 * db_idx], 0, 16, 112,
                               3024, 0)
        with tik_instance.for_range(0, 112) as cc7, tik_instance.for_range(0, 16) as cc8:
            tik_instance.vadds(16, t_transpose_local_ub[cc7 * 256 + cc8 * 16],
                               input_1_local_ub[1792 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 802816 + cc1_db * 57344 + 28672 * db_idx],
                               t_transpose_local_ub, 0, 1, 1792, 0, 0)
    return tik_instance, res


def shape3(tik_instance, input_x, res, dtype):
    """input shape (32, 8, 56, 56, 16)"""
    zero = tik_instance.Scalar(dtype="float16", init_value=0)
    with tik_instance.for_range(0, 32, block_num=32) as block_idx, tik_instance.for_range(0, 7) as cc1_db, \
            tik_instance.for_range(0, 2, thread_num=2) as db_idx:
        input_1_local_ub = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub",
                                               scope=tik.scope_ubuf)
        t_transpose_local_ub = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub",
                                                   scope=tik.scope_ubuf)
        tik_instance.data_move(input_1_local_ub,
                               input_x[block_idx * 401408 + cc1_db * 7168 + 3584 * db_idx], 0, 8, 224,
                               2912, 0)
        with tik_instance.for_range(0, 224) as cc7, tik_instance.for_range(0, 16) as cc8:
            tik_instance.vadds(16, t_transpose_local_ub[cc7 * 128 + cc8 * 16],
                               input_1_local_ub[3584 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 401408 + cc1_db * 57344 + 28672 * db_idx],
                               t_transpose_local_ub, 0, 1, 1792, 0, 0)
    return tik_instance, res


def shape4(tik_instance, input_x, res, dtype):
    """input shape (32, 8, 28, 28, 16)"""
    zero = tik_instance.Scalar(dtype="float16", init_value=0)
    with tik_instance.for_range(0, 32, block_num=32) as block_idx, tik_instance.for_range(0, 2) as cc1_db, \
            tik_instance.for_range(0, 2, thread_num=2) as db_idx:
        input_1_local_ub = tik_instance.Tensor(dtype, [25088], name="input_1_local_ub",
                                               scope=tik.scope_ubuf)
        t_transpose_local_ub = tik_instance.Tensor(dtype, [25088], name="t_transpose_local_ub",
                                                   scope=tik.scope_ubuf)
        tik_instance.data_move(input_1_local_ub,
                               input_x[block_idx * 100352 + cc1_db * 6272 + 3136 * db_idx], 0, 8, 196, 588,
                               0)
        with tik_instance.for_range(0, 196) as cc7, tik_instance.for_range(0, 8) as cc8:
            tik_instance.vadds(16, t_transpose_local_ub[cc7 * 128 + cc8 * 16],
                               input_1_local_ub[3136 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 100352 + cc1_db * 50176 + 25088 * db_idx],
                               t_transpose_local_ub, 0, 1, 1568, 0, 0)
    return tik_instance, res


def shape5(tik_instance, input_x, res, dtype):
    """input shape (32, 32, 28, 28, 16)"""
    zero = tik_instance.Scalar(dtype="float16", init_value=0)
    with tik_instance.for_range(0, 32, block_num=32) as block_idx, tik_instance.for_range(0, 7) as cc1_db, \
            tik_instance.for_range(0, 2, thread_num=2) as db_idx:
        input_1_local_ub = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub",
                                               scope=tik.scope_ubuf)
        t_transpose_local_ub = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub",
                                                   scope=tik.scope_ubuf)
        tik_instance.data_move(input_1_local_ub, input_x[block_idx * 401408 + cc1_db * 1792 +
                                                         896 * db_idx], 0, 32, 56, 728, 0)
        with tik_instance.for_range(0, 56) as cc7, tik_instance.for_range(0, 32) as cc8:
            tik_instance.vadds(16, t_transpose_local_ub[cc7 * 512 + cc8 * 16],
                               input_1_local_ub[896 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 401408 + cc1_db * 57344 + 28672 * db_idx],
                               t_transpose_local_ub, 0, 1, 1792, 0, 0)
    return tik_instance, res


def shape6(tik_instance, input_x, res, dtype):
    """input shape (32, 16, 28, 28, 16)"""
    with tik_instance.for_range(0, 32, block_num=32) as block_idx:
        zero = tik_instance.Scalar(dtype="float16", init_value=0)
        with tik_instance.for_range(0, 3) as cc1_db, tik_instance.for_range(0, 2, thread_num=2) as db_idx:
            input_1_local_ub = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub",
                                                   scope=tik.scope_ubuf)
            t_transpose_local_ub = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_1_local_ub,
                                   input_x[block_idx * 200704 + cc1_db * 3584 + 1792 * db_idx], 0, 16, 112,
                                   672, 0)
            with tik_instance.for_range(0, 112) as cc7, tik_instance.for_range(0, 16) as cc8:
                tik_instance.vadds(16, t_transpose_local_ub[cc7 * 256 + cc8 * 16],
                                   input_1_local_ub[1792 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
            tik_instance.data_move(res[block_idx * 200704 + cc1_db * 57344 + 28672 * db_idx],
                                   t_transpose_local_ub, 0, 1, 1792, 0, 0)

        input_1_local_ub2 = tik_instance.Tensor(dtype, [28672], name="input_1_local_ub2", scope=tik.scope_ubuf)
        t_transpose_local_ub2 = tik_instance.Tensor(dtype, [28672], name="t_transpose_local_ub2",
                                                    scope=tik.scope_ubuf)
        tik_instance.data_move(input_1_local_ub2, input_x[block_idx * 200704 + 10752], 0, 16, 112, 672, 0)
        with tik_instance.for_range(0, 112) as cc7, tik_instance.for_range(0, 16) as cc8:
            tik_instance.vadds(16, t_transpose_local_ub2[cc7 * 256 + cc8 * 16],
                               input_1_local_ub2[1792 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 200704 + 172032], t_transpose_local_ub2, 0, 1, 1792, 0, 0)
    return tik_instance, res


def shape7(tik_instance, input_x, res, dtype):
    """input shape (32, 16, 14, 14, 16)"""
    with tik_instance.for_range(0, 32, block_num=32) as block_idx:
        zero = tik_instance.Scalar(dtype="float16", init_value=0)
        with tik_instance.for_range(0, 2, thread_num=2) as db_idx:
            input_1_local_ub = tik_instance.Tensor(dtype, [25088], name="input_1_local_ub", scope=tik.scope_ubuf)
            t_transpose_local_ub = tik_instance.Tensor(dtype, [25088], name="t_transpose_local_ub",
                                                       scope=tik.scope_ubuf)
            tik_instance.data_move(input_1_local_ub, input_x[block_idx * 50176 + 1568 * db_idx], 0, 16, 98, 98, 0)
            with tik_instance.for_range(0, 98) as cc7, tik_instance.for_range(0, 16) as cc8:
                tik_instance.vadds(16, t_transpose_local_ub[cc7 * 256 + cc8 * 16],
                                   input_1_local_ub[1568 * cc8 + cc7 * 16], zero, 1, 1, 1, 0, 0)
            tik_instance.data_move(res[block_idx * 50176 + 25088 * db_idx], t_transpose_local_ub, 0, 1, 1568, 0, 0)
    return tik_instance, res


def shape8(tik_instance, input_x, res, dtype):
    """input shape (32, 128, 7, 7, 16)"""
    with tik_instance.for_range(0, 32, block_num=32) as block_idx, tik_instance.for_range(0, 7, thread_num=2) \
            as cc1:
        input_x_ub = tik_instance.Tensor(dtype, [1, 128, 1, 7, 16], name="input_1_local_ub",
                                         scope=tik.scope_ubuf)
        transpose_ub = tik_instance.Tensor(dtype, [1, 1, 7, 128, 16], name="transpose_local_UB",
                                           scope=tik.scope_ubuf)
        tik_instance.data_move(input_x_ub, input_x[block_idx, 0, cc1, 0, 0], 0, 128, 7, 42, 0)
        with tik_instance.for_range(0, 7) as cc7, tik_instance.for_range(0, 128) as cc8:
            tik_instance.vadds(16, transpose_ub[0, 0, cc7, cc8, 0], input_x_ub[0, cc8, 0, cc7, 0], 0,
                               1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 100352 + 14336 * cc1], transpose_ub, 0, 1, 896, 0, 0)
    return tik_instance, res


def shape9(tik_instance, input_x, res, dtype):
    """input shape (32, 32, 7, 7, 16)"""
    with tik_instance.for_range(0, 32, block_num=32) as block_idx:
        input_x_ub = tik_instance.Tensor(dtype, [1, 32, 7, 7, 16], name="input_1_local_ub",
                                         scope=tik.scope_ubuf)
        transpose_ub = tik_instance.Tensor(dtype, [1, 7, 7, 32, 16], name="transpose_local_UB",
                                           scope=tik.scope_ubuf)
        tik_instance.data_move(input_x_ub, input_x[block_idx, 0, 0, 0, 0], 0, 1, 1568, 0, 0)
        with tik_instance.for_range(0, 7) as cc1, tik_instance.for_range(0, 7) as cc2, \
                tik_instance.for_range(0, 32) as cc3:
            tik_instance.vadds(16, transpose_ub[0, cc1, cc2, cc3, 0], input_x_ub[0, cc3, cc1, cc2, 0], 0,
                               1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 25088], transpose_ub, 0, 1, 1568, 0, 0)
    return tik_instance, res


def shape10(tik_instance, input_x, res, dtype):
    """input shape (32, 32, 14, 14, 16)"""

    def _inner_compute(split_index):
        input_x_ub = tik_instance.Tensor(dtype, [1, 32, 2, 14, 16], name="input_1_local_ub",
                                         scope=tik.scope_ubuf)
        transpose_ub = tik_instance.Tensor(dtype, [1, 2, 14, 32, 16], name="transpose_local_UB",
                                           scope=tik.scope_ubuf)
        tik_instance.data_move(input_x_ub, input_x[block_idx, 0, split_index * 2, 0, 0], 0, 32, 28, 168, 0)
        with tik_instance.for_range(0, 2) as cc2, tik_instance.for_range(0, 14) as cc3, \
                tik_instance.for_range(0, 32) as cc4:
            tik_instance.vadds(16, transpose_ub[0, cc2, cc3, cc4, 0], input_x_ub[0, cc4, cc2, cc3, 0],
                               0, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 100352 + split_index * 2 * 7168], transpose_ub, 0, 1, 896, 0, 0)

    with tik_instance.for_range(0, 32, block_num=32) as block_idx:
        with tik_instance.for_range(0, 6, thread_num=2) as cc1:
            _inner_compute(cc1)
        _inner_compute(6)
    return tik_instance, res


def shape11(tik_instance, input_x, res, dtype):
    """input shape (32, 64, 14, 14, 16)"""

    def _inner_compute(split_index, block_idx):
        input_x_ub = tik_instance.Tensor(dtype, [1, 64, 2, 14, 16], name="input_1_local_ub",
                                         scope=tik.scope_ubuf)
        transpose_ub = tik_instance.Tensor(dtype, [1, 2, 14, 64, 16], name="transpose_local_UB",
                                           scope=tik.scope_ubuf)
        tik_instance.data_move(input_x_ub, input_x[block_idx, 0, split_index * 2, 0, 0], 0, 64, 28, 168, 0)
        with tik_instance.for_range(0, 2) as cc2, tik_instance.for_range(0, 14) as cc3, \
                tik_instance.for_range(0, 64) as cc4:
            tik_instance.vadds(16, transpose_ub[0, cc2, cc3, cc4, 0], input_x_ub[0, cc4, cc2, cc3, 0],
                               0, 1, 1, 1, 0, 0)
        tik_instance.data_move(res[block_idx * 200704 + split_index * 2 * 14336], transpose_ub, 0, 1, 1792, 0, 0)

    with tik_instance.for_range(0, 32, block_num=32) as block_idx:
        with tik_instance.for_range(0, 6, thread_num=2) as cc1:
            _inner_compute(cc1, block_idx)
        _inner_compute(6, block_idx)
    return tik_instance, res
