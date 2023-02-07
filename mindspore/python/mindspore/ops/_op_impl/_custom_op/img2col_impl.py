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
"""CusImg2ColNC1HWC0"""
from __future__ import absolute_import

from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

cus_img2col_info = TBERegOp("CusImg2Col") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("img2col.so") \
    .compute_cost(10) \
    .kernel_name("cus_img2col") \
    .partial_flag(True) \
    .attr("ksizes", "required", "listInt", "all") \
    .attr("strides", "required", "listInt", "all") \
    .attr("dilates", "required", "listInt", "all") \
    .attr("mode", "required", "str", "all") \
    .input(0, "x1", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_5HD, DataType.F16_FracNZ) \
    .get_op_info()


def shape56_0(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 4, 56, 56, 16), 'float16', (3, 3), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [1, 1, 1, 1]
    l1_h, l1_w, jump_stride, repeat_mode = 56, 56, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (200704,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (50176,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 0, 0, 0], 0, 1, 12544, 0, 0)
        with tik_instance.for_range(0, 9) as eeb0, tik_instance.for_range(0, 4) as eeb1:
            rep = 196
            fetch_filter_w = eeb0 % 3
            fetch_filter_h = eeb0 // 3
            left_top_w, left_top_h = -1, -1
            c1_index = eeb1
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub, input_1_1_local_l1, pad,
                                  l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h, left_top_w,
                                  left_top_h, stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                                  dilation_filter_h, jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, rep) as i:
                tik_instance.data_move(res[eeb1 * 9 + eeb0, i + 196 * block_index, 0, 0],
                                       input_1_1_fractal_l1_local_ub[i * 256], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape56_1(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 8, 56, 56, 16), 'float16', (3, 3), (2, 2))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [1, 1, 1, 1]
    l1_h, l1_w, jump_stride, repeat_mode = 56, 56, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (401408,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (112896,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 0, 0, 0], 0, 1, 25088, 0, 0)
        with tik_instance.for_range(0, 8) as eeb0, tik_instance.for_range(0, 9) as eeb1:
            rep = 49
            fetch_filter_w = eeb1 % 3
            fetch_filter_h = eeb1 // 3
            left_top_w, left_top_h = -1, -1
            c1_index = eeb0
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[49 * 256 * eeb1], input_1_1_local_l1,
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h, left_top_w,
                                  left_top_h, stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                                  dilation_filter_h, jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 9) as eeb1, tik_instance.for_range(0, 49) as i:
                tik_instance.data_move(res[eeb1 + eeb0 * 9, 49 * block_index + i, 0, 0],
                                       input_1_1_fractal_l1_local_ub[i * 256 + eeb1 * 49 * 256],
                                       0, 1, 16, 0, 0)
    return tik_instance, res


def shape56_2(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 4, 56, 56, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, c1_index, jump_stride, repeat_mode = 56, 56, 0, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (12544 * 32 // 2,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (100352 // 2,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 0, 0, 0], 0, 1, 12544, 0, 0)
        with tik_instance.for_range(0, 4) as eeb:
            fetch_filter_w, fetch_filter_h, left_top_h, left_top_w = 0, 0, 0, 0
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub, input_1_1_local_l1[eeb * 56 * 56 * 16],
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                  left_top_w, left_top_h, stride_w, stride_h,
                                  filter_w, filter_h, dilation_filter_w, dilation_filter_h, jump_stride,
                                  repeat_mode, 196)
            with tik_instance.for_range(0, 196) as rep:
                tik_instance.data_move(res[eeb, rep + block_index * 196, 0, 0],
                                       input_1_1_fractal_l1_local_ub[rep * 256], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape56_3(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 16, 56, 56, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, c1_index, jump_stride, repeat_mode = 56, 56, 0, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (25088 * 32 // 2,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (196 * 256 * 2,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        with tik_instance.for_range(0, 2) as eeb0, \
                tik_instance.for_range(0, 4) as eeb1, \
                tik_instance.for_range(0, 2) as eeb2:
            tik_instance.data_move(input_1_1_local_l1, input_x[block_index, eeb0 * 8, 0, 0, 0], 0, 1, 25088, 0, 0)
            fetch_filter_w, fetch_filter_h, left_top_h, left_top_w = 0, 0, 0, 0
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[eeb2 * 196 * 256],
                                  input_1_1_local_l1[(eeb2 + eeb1 * 2) * 56 * 56 * 16],
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                  left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                  filter_h, dilation_filter_w, dilation_filter_h,
                                  jump_stride, repeat_mode, 196)
            with tik_instance.for_range(0, 2) as eeb2, tik_instance.for_range(0, 196) as i:
                tik_instance.data_move(res[eeb0 * 8 + eeb1 * 2 + eeb2, i + block_index * 196, 0, 0],
                                       input_1_1_fractal_l1_local_ub[256 * i + eeb2 * 196 * 256], 0, 1, 16,
                                       0, 0)
    return tik_instance, res


def shape56_4(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 16, 56, 56, 16), 'float16', (1, 1), (2, 2))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, c1_index, jump_stride, repeat_mode = 56, 56, 0, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index, \
            tik_instance.for_range(0, 2) as eeb0, \
            tik_instance.for_range(0, 8) as eeb1:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (25088 * 32 // 2,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (49 * 256 * 8,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, eeb0 * 8, 0, 0, 0], 0, 1, 25088, 0, 0)
        fetch_filter_w, fetch_filter_h, left_top_h, left_top_w = 0, 0, 0, 0
        tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[eeb1 * 49 * 256],
                              input_1_1_local_l1[eeb1 * 56 * 56 * 16],
                              pad, l1_h, l1_w, c1_index, fetch_filter_w,
                              fetch_filter_h, left_top_w, left_top_h,
                              stride_w, stride_h, filter_w, filter_h,
                              dilation_filter_w, dilation_filter_h,
                              jump_stride, repeat_mode, 49)
        with tik_instance.for_range(0, 8) as eeb1, \
                tik_instance.for_range(0, 49) as i:
            tik_instance.data_move(res[eeb0 * 8 + eeb1, i + block_index * 49, 0, 0],
                                   input_1_1_fractal_l1_local_ub[256 * i + eeb1 * 49 * 256],
                                   0, 1, 16, 0, 0)
    return tik_instance, res


def shape28_0(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 8, 28, 28, 16), 'float16', (3, 3), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [1, 1, 1, 1]
    l1_h, l1_w, jump_stride, repeat_mode = 28, 28, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (100352,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (112896,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 0, 0, 0], 0, 1, 6272, 0, 0)
        with tik_instance.for_range(0, 8) as eeb0, tik_instance.for_range(0, 9) as eeb1:
            rep = 49
            c1_index = 0
            fetch_filter_w = eeb1 % 3
            fetch_filter_h = eeb1 // 3
            left_top_w, left_top_h = -1, -1
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[49 * 256 * eeb1], input_1_1_local_l1, pad,
                                  l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h, left_top_w, left_top_h,
                                  stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h,
                                  jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 9) as eeb1, tik_instance.for_range(0, 49) as i:
                tik_instance.data_move(res[eeb1 + eeb0 * 9, 49 * block_index + i, 0, 0],
                                       input_1_1_fractal_l1_local_ub[i * 256 + eeb1 * 49 * 256],
                                       0, 1, 16, 0, 0)
    return tik_instance, res


def shape28_1(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 16, 28, 28, 16), 'float16', (3, 3), (2, 2))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [1, 1, 1, 1]
    l1_h, l1_w, jump_stride, repeat_mode = 28, 28, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        eeb0 = block_index % 2
        eeb1 = block_index // 2
        input_1_1_local_l1 = tik_instance.Tensor("float16", (200704,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (53248,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (50176,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        with tik_instance.for_range(0, 16) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 12544], input_x[i + 16 * eeb0, eeb1, 0, 0, 0], 0, 1, 784,
                                   0, 0)

        with tik_instance.for_range(0, 9) as eeb3:
            rep = 13
            fetch_filter_w = eeb3 % 3
            fetch_filter_h = eeb3 // 3
            left_top_w, left_top_h = -1, -1
            c1_index = 0
            with tik_instance.for_range(0, 16) as i:
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[3328 * i], input_1_1_local_l1[12544 * i],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h, left_top_w,
                                      left_top_h, stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                                      dilation_filter_h, jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 16) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 196 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 3328], 0, 1, 196, 0, 0)

            with tik_instance.for_range(196 * eeb0, 196 * (eeb0 + 1)) as i:
                tik_instance.data_move(res[eeb1 * 9 + eeb3, i, 0, 0],
                                       input_1_2_fractal_l1_local_ub[256 * (i - 196 * eeb0)], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape28_2(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 32, 28, 28, 16), 'float16', (1, 1), (2, 2))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, jump_stride, repeat_mode = 28, 28, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (401408,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (53248,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (50176,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        with tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 12544], input_x[i, block_index, 0, 0, 0],
                                   0, 1, 784, 0, 0)
        with tik_instance.for_range(0, 16) as i:
            rep = 13
            fetch_filter_w, fetch_filter_h, left_top_w, left_top_h = 0, 0, 0, 0
            c1_index = 0
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[3328 * i], input_1_1_local_l1[12544 * i],
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                  left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                  filter_h, dilation_filter_w, dilation_filter_h,
                                  jump_stride, repeat_mode, rep)
        with tik_instance.for_range(0, 16) as i:
            tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 196 * 16],
                                   input_1_1_fractal_l1_local_ub[i * 3328], 0, 1, 196, 0, 0)
        with tik_instance.for_range(0, 196) as i:
            tik_instance.data_move(res[block_index, i, 0, 0], input_1_2_fractal_l1_local_ub[256 * i], 0, 1, 16, 0,
                                   0)

        with tik_instance.for_range(16, 32) as i:
            rep = 13
            fetch_filter_w, fetch_filter_h, left_top_w, left_top_h = 0, 0, 0, 0
            c1_index = 0
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[3328 * (i - 16)], input_1_1_local_l1[12544 * i],
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h, left_top_w,
                                  left_top_h, stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                                  dilation_filter_h, jump_stride, repeat_mode, rep)
        with tik_instance.for_range(0, 16) as i:
            tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 196 * 16],
                                   input_1_1_fractal_l1_local_ub[i * 3328], 0, 1, 196, 0, 0)
        with tik_instance.for_range(196, 392) as i:
            tik_instance.data_move(res[block_index, i, 0, 0], input_1_2_fractal_l1_local_ub[256 * (i - 196)], 0, 1,
                                   16, 0, 0)
    return tik_instance, res


def shape28_3(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 8, 28, 28, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, jump_stride, repeat_mode = 28, 28, 1, 1
    c1_index = 0
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (6272 * 32 // 2,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (49 * 256 * 8,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 0, 0, 0], 0, 1, 6272, 0, 0)
        with tik_instance.for_range(0, 1) as eeb0:
            with tik_instance.for_range(0, 8) as eeb1:
                fetch_filter_w, fetch_filter_h, left_top_w, left_top_h = 0, 0, 0, 0
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[eeb1 * 49 * 256],
                                      input_1_1_local_l1[(eeb1 + eeb0 * 8) * 28 * 28 * 16],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h,
                                      filter_w, filter_h, dilation_filter_w, dilation_filter_h,
                                      jump_stride, repeat_mode, 49)
            with tik_instance.for_range(0, 8) as eeb1, tik_instance.for_range(0, 49) as i:
                tik_instance.data_move(res[eeb0 * 8 + eeb1, i + block_index * 49, 0, 0],
                                       input_1_1_fractal_l1_local_ub[i * 256 + eeb1 * 49 * 256],
                                       0, 1, 16, 0, 0)
    return tik_instance, res


def shape28_4(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 32, 28, 28, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, jump_stride, repeat_mode = 28, 28, 1, 1
    c1_index = 0
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (25088 * 32 // 2,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (49 * 256 * 8,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 0, 0, 0], 0, 1, 25088, 0, 0)
        with tik_instance.for_range(0, 4) as eeb0:
            with tik_instance.for_range(0, 8) as eeb1:
                fetch_filter_w, fetch_filter_h, left_top_w, left_top_h = 0, 0, 0, 0
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[eeb1 * 49 * 256],
                                      input_1_1_local_l1[(eeb1 + eeb0 * 8) * 28 * 28 * 16],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                      filter_h, dilation_filter_w, dilation_filter_h, jump_stride,
                                      repeat_mode, 49)
            with tik_instance.for_range(0, 8) as eeb1, tik_instance.for_range(0, 49) as i:
                tik_instance.data_move(res[eeb0 * 8 + eeb1, i + block_index * 49, 0, 0],
                                       input_1_1_fractal_l1_local_ub[i * 256 + eeb1 * 49 * 256],
                                       0, 1, 16, 0, 0)
    return tik_instance, res


def shape14_0(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 16, 14, 14, 16), 'float16', (3, 3), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [1, 1, 1, 1]
    l1_h, l1_w, jump_stride, repeat_mode = 14, 14, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        eeb0 = block_index % 2
        eeb1 = block_index // 2
        input_1_1_local_l1 = tik_instance.Tensor("float16", (50176,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (53248,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (50176,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        with tik_instance.for_range(0, 16) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 3136], input_x[i + 16 * eeb0, eeb1, 0, 0, 0], 0, 1, 196,
                                   0, 0)

        with tik_instance.for_range(0, 9) as eeb3:
            rep = 13
            fetch_filter_w = eeb3 % 3
            fetch_filter_h = eeb3 // 3
            left_top_w, left_top_h = -1, -1
            c1_index = 0
            with tik_instance.for_range(0, 16) as i:
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[3328 * i], input_1_1_local_l1[3136 * i],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h,
                                      filter_w, filter_h, dilation_filter_w, dilation_filter_h,
                                      jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 16) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 196 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 3328], 0, 1, 196, 0, 0)

            with tik_instance.for_range(196 * eeb0, 196 * (eeb0 + 1)) as i:
                tik_instance.data_move(res[eeb1 * 9 + eeb3, i, 0, 0],
                                       input_1_2_fractal_l1_local_ub[256 * (i - 196 * eeb0)], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape14_1(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 32, 14, 14, 16), 'float16', (3, 3), (2, 2))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [1, 1, 1, 1]
    l1_h, l1_w, jump_stride, repeat_mode = 14, 14, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (100352,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (32768,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (25088,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        with tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 3136], input_x[i, block_index, 0, 0, 0], 0, 1, 196, 0, 0)
        with tik_instance.for_range(0, 9) as eeb:
            rep = 4
            fetch_filter_w = eeb % 3
            fetch_filter_h = eeb // 3
            left_top_w, left_top_h = -1, -1
            c1_index = 0
            with tik_instance.for_range(0, 32) as i:
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[1024 * i], input_1_1_local_l1[3136 * i],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                      filter_h, dilation_filter_w, dilation_filter_h,
                                      jump_stride, repeat_mode, rep)

            with tik_instance.for_range(0, 32) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 49 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 1024], 0, 1, 49, 0, 0)

            with tik_instance.for_range(0, 98) as i:
                tik_instance.data_move(res[eeb + block_index * 9, i, 0, 0], input_1_2_fractal_l1_local_ub[256 * i],
                                       0, 1, 16, 0, 0)
    return tik_instance, res


def shape14_2(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 64, 14, 14, 16), 'float16', (1, 1), (2, 2))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, jump_stride, repeat_mode = 14, 14, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (100352,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (32768,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (25088,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")

        with tik_instance.for_range(0, 2) as eeb0, tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 3136], input_x[i, block_index * 2 + eeb0, 0, 0, 0],
                                   0, 1, 196, 0, 0)
            with tik_instance.for_range(0, 32) as i:
                rep = 4
                fetch_filter_w, fetch_filter_h, left_top_w, left_top_h = 0, 0, 0, 0
                c1_index = 0
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[1024 * i], input_1_1_local_l1[3136 * i],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                      filter_h, dilation_filter_w, dilation_filter_h,
                                      jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 32) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 49 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 1024], 0, 1, 49, 0, 0)

            with tik_instance.for_range(0, 98) as i:
                tik_instance.data_move(res[eeb0 + block_index * 2, i, 0, 0],
                                       input_1_2_fractal_l1_local_ub[256 * i], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape14_3(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 64, 14, 14, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, jump_stride, repeat_mode = 14, 14, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (100352,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_2_local_l1 = tik_instance.Tensor("float16", (100352,), scope=tik.scope_cbuf,
                                                 name="input_1_2_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (53248,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (50176,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        with tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 3136], input_x[i, block_index * 2, 0, 0, 0],
                                   0, 1, 196, 0, 0)
            tik_instance.data_move(input_1_2_local_l1[i * 3136], input_x[i, block_index * 2 + 1, 0, 0, 0], 0, 1,
                                   196, 0, 0)
        with tik_instance.for_range(0, 2) as eeb1:
            with tik_instance.for_range(eeb1 * 16, (eeb1 + 1) * 16) as i:
                rep = 13
                c1_index = 0
                fetch_filter_w, fetch_filter_h, left_top_w, left_top_h, rep, c1_index = 0, 0, 0, 0, 13, 0
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[3328 * (i - eeb1 * 16)],
                                      input_1_1_local_l1[3136 * i],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                      filter_h, dilation_filter_w, dilation_filter_h,
                                      jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 16) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 196 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 3328], 0, 1, 196, 0, 0)
            with tik_instance.for_range(eeb1 * 196, (eeb1 + 1) * 196) as i:
                tik_instance.data_move(res[block_index * 2, i, 0, 0],
                                       input_1_2_fractal_l1_local_ub[256 * (i - eeb1 * 196)], 0, 1, 16, 0, 0)

        with tik_instance.for_range(0, 2) as eeb1:
            with tik_instance.for_range(eeb1 * 16, (eeb1 + 1) * 16) as i:
                fetch_filter_w, fetch_filter_h, left_top_w, left_top_h, rep, c1_index = 0, 0, 0, 0, 13, 0
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[3328 * (i - eeb1 * 16)],
                                      input_1_2_local_l1[3136 * i],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                      filter_h, dilation_filter_w, dilation_filter_h,
                                      jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 16) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 196 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 3328], 0, 1, 196, 0, 0)
            with tik_instance.for_range(eeb1 * 196, (eeb1 + 1) * 196) as i:
                tik_instance.data_move(res[block_index * 2 + 1, i, 0, 0],
                                       input_1_2_fractal_l1_local_ub[256 * (i - eeb1 * 196)], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape14_4(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 16, 14, 14, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h = 14
    l1_w = 14
    c1_index = 0
    jump_stride = 1
    repeat_mode = 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        eeb0 = block_index % 2
        eeb1 = block_index // 2
        input_1_1_local_l1 = tik_instance.Tensor("float16", (196 * 32 * 16,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (106496 // 2,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (196 * 16 * 16,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        with tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 3136], input_x[i, eeb1, 0, 0, 0], 0, 1, 196, 0, 0)
        with tik_instance.for_range(0, 16) as i:
            fetch_filter_w = 0
            fetch_filter_h = 0
            left_top_h = 0
            left_top_w = 0
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[i * 3328],
                                  input_1_1_local_l1[i * 3136 + eeb0 * 16 * 3136],
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                  left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                  filter_h, dilation_filter_w, dilation_filter_h,
                                  jump_stride, repeat_mode, 13)
        with tik_instance.for_range(0, 16) as i:
            tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 196 * 16],
                                   input_1_1_fractal_l1_local_ub[i * 3328], 0, 1, 196, 0, 0)
        with tik_instance.for_range(0, 196) as i:
            tik_instance.data_move(res[eeb1, i + 196 * eeb0, 0, 0], input_1_2_fractal_l1_local_ub[256 * i], 0, 1,
                                   16, 0, 0)
    return tik_instance, res


def shape7_0(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 32, 7, 7, 16), 'float16', (3, 3), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [1, 1, 1, 1]
    l1_h = 7
    l1_w = 7
    jump_stride = 1
    repeat_mode = 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (25088,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (32768,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (25088,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        with tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 784], input_x[i, block_index, 0, 0, 0], 0, 1, 49, 0, 0)

        with tik_instance.for_range(0, 9) as eeb:
            rep = 4
            fetch_filter_w = eeb % 3
            fetch_filter_h = eeb // 3
            left_top_w = -1
            left_top_h = -1
            c1_index = 0
            with tik_instance.for_range(0, 32) as i:
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[1024 * i],
                                      input_1_1_local_l1[784 * i], pad, l1_h, l1_w, c1_index,
                                      fetch_filter_w, fetch_filter_h, left_top_w, left_top_h,
                                      stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                                      dilation_filter_h, jump_stride, repeat_mode, rep)
            with tik_instance.for_range(0, 32) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 49 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 1024], 0, 1, 49, 0, 0)

            with tik_instance.for_range(0, 98) as i:
                tik_instance.data_move(res[eeb + block_index * 9, i, 0, 0],
                                       input_1_2_fractal_l1_local_ub[256 * i], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape7_1(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 128, 7, 7, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h, l1_w, jump_stride, repeat_mode = 7, 7, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (25088,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        input_1_1_local_l1 = tik_instance.Tensor("float16", (25088,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (32768,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")

        with tik_instance.for_range(0, 4) as eeb0, tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 784], input_x[i, eeb0 + block_index * 4, 0, 0, 0], 0,
                                   1, 49, 0, 0)
            with tik_instance.for_range(0, 32) as i:
                rep = 4
                fetch_filter_w = 0
                fetch_filter_h = 0
                left_top_w = 0
                left_top_h = 0
                c1_index = 0
                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[1024 * i], input_1_1_local_l1[784 * i],
                                      pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                      left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                      filter_h, dilation_filter_w, dilation_filter_h,
                                      jump_stride, repeat_mode, rep)

            with tik_instance.for_range(0, 32) as i:
                tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 49 * 16],
                                       input_1_1_fractal_l1_local_ub[i * 1024], 0, 1, 49, 0, 0)

            with tik_instance.for_range(0, 98) as i:
                tik_instance.data_move(res[eeb0 + block_index * 4, i, 0, 0],
                                       input_1_2_fractal_l1_local_ub[256 * i], 0, 1, 16, 0, 0)
    return tik_instance, res


def shape7_2(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 32, 7, 7, 16), 'float16', (1, 1), (1, 1))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [0, 0, 0, 0]
    l1_h = 7
    l1_w = 7
    c1_index = 0
    jump_stride, repeat_mode = 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_2_fractal_l1_local_ub = tik_instance.Tensor("float16", (25088,), scope=tik.scope_ubuf,
                                                            name="input_1_2_fractal_l1_local_ub")
        input_1_1_local_l1 = tik_instance.Tensor("float16", (25088,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (32768,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")


        with tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_1_local_l1[i * 784], input_x[i, block_index, 0, 0, 0], 0, 1, 49, 0, 0)

        with tik_instance.for_range(0, 32) as i:
            fetch_filter_w = 0
            fetch_filter_h = 0
            left_top_h = 0
            left_top_w = 0
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub[1024 * i], input_1_1_local_l1[784 * i],
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                  left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                  filter_h, dilation_filter_w, dilation_filter_h,
                                  jump_stride, repeat_mode, 4)

        with tik_instance.for_range(0, 32) as i:
            tik_instance.data_move(input_1_2_fractal_l1_local_ub[i * 49 * 16],
                                   input_1_1_fractal_l1_local_ub[i * 1024], 0, 1, 49, 0, 0)
        with tik_instance.for_range(0, 98) as i:
            tik_instance.data_move(res[block_index, i, 0, 0], input_1_2_fractal_l1_local_ub[i * 256], 0, 1, 16, 0,
                                   0)
    return tik_instance, res


def height224_width224(tik_instance, input_x, res, input_shape, shape_info):
    """input_shape is ((32, 1, 224, 224, 16), 'float16', (7, 7), (2, 2))"""
    stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h = shape_info
    pad = [3, 3, 3, 3]
    l1_h, l1_w, c1_index, jump_stride, repeat_mode = 56, 224, 0, 1, 1
    with tik_instance.for_range(0, 32, block_num=32) as block_index:
        input_1_1_local_l1 = tik_instance.Tensor("float16", (200704,), scope=tik.scope_cbuf,
                                                 name="input_1_1_local_l1")
        input_1_1_fractal_l1_local_ub = tik_instance.Tensor("float16", (53760,), scope=tik.scope_ubuf,
                                                            name="input_1_1_fractal_l1_local_ub")

        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 0, 0, 0], 0, 1, 12544, 0, 0)
        with tik_instance.for_range(0, 7) as eeb, tik_instance.for_range(0, 7) as cc0:
            temp = eeb % 2
            rep = ((55 - temp - (-3 + eeb)) // 2 + 1) * 7
            fetch_filter_w = cc0
            fetch_filter_h = eeb
            left_top_w = -3
            left_top_h = -3

            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub, input_1_1_local_l1,
                                  pad, l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                  left_top_w, left_top_h, stride_w, stride_h, filter_w, filter_h,
                                  dilation_filter_w, dilation_filter_h, jump_stride, repeat_mode, rep)

            with tik_instance.for_range(0, rep) as cc1:
                tik_instance.data_move(res[cc0 + eeb * 7, cc1 + 784 * block_index, 0, 0],
                                       input_1_1_fractal_l1_local_ub[cc1 * 256], 0, 1, 16, 0, 0)

        with tik_instance.for_range(1, 3) as eeb0:
            tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 56 * eeb0, 0, 0], 0, 1, 12544, 0, 0)
            with tik_instance.for_range(0, 7) as eeb, tik_instance.for_range(0, 7) as cc0:
                temp = eeb % 2
                rep_prefix = ((55 - temp - (-3 + eeb)) // 2 + 1) * 7
                rep = 196
                fetch_filter_w = cc0
                fetch_filter_h = eeb
                left_top_w = -3

                left_top_h = 1 + ((55 - temp - (-3 + eeb)) // 2 - 29) * 2

                tik_instance.load3dv1(input_1_1_fractal_l1_local_ub, input_1_1_local_l1, pad, l1_h, l1_w,
                                      c1_index, fetch_filter_w, fetch_filter_h, left_top_w, left_top_h,
                                      stride_w, stride_h, filter_w, filter_h, dilation_filter_w,
                                      dilation_filter_h, jump_stride, repeat_mode, rep)
                with tik_instance.for_range(0, rep) as cc1:
                    tik_instance.data_move(
                        res[cc0 + eeb * 7, cc1 + rep_prefix + (eeb0 - 1) * rep + 784 * block_index, 0, 0],
                        input_1_1_fractal_l1_local_ub[cc1 * 256], 0, 1, 16, 0, 0)

        tik_instance.data_move(input_1_1_local_l1, input_x[block_index, 0, 56 * 3, 0, 0], 0, 1, 12544, 0, 0)

        with tik_instance.for_range(0, 7) as eeb, tik_instance.for_range(0, 7) as cc0:
            temp = eeb % 2
            rep_prefix = ((55 - temp - (-3 + eeb)) // 2 + 1) * 7 + 196 * 2
            rep = 784 - rep_prefix
            fetch_filter_w = cc0
            fetch_filter_h = eeb
            left_top_w = -3
            left_top_h = 1 + ((55 - temp - (-3 + eeb)) // 2 - 29) * 2
            tik_instance.load3dv1(input_1_1_fractal_l1_local_ub, input_1_1_local_l1, pad,
                                  l1_h, l1_w, c1_index, fetch_filter_w, fetch_filter_h,
                                  left_top_w, left_top_h, stride_w, stride_h, filter_w,
                                  filter_h, dilation_filter_w, dilation_filter_h, jump_stride,
                                  repeat_mode, rep)

            with tik_instance.for_range(0, rep) as cc1:
                tik_instance.data_move(res[cc0 + eeb * 7, cc1 + rep_prefix + 784 * block_index, 0, 0],
                                       input_1_1_fractal_l1_local_ub[cc1 * 256], 0, 1, 16, 0, 0)
    return tik_instance, res


def height56_width56(tik_instance, input_x, res, input_shape, shape_info):
    """height56_width56"""
    if input_shape == ((32, 4, 56, 56, 16), 'float16', (3, 3), (1, 1)):
        tik_instance, res = shape56_0(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 8, 56, 56, 16), 'float16', (3, 3), (2, 2)):
        tik_instance, res = shape56_1(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 4, 56, 56, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape56_2(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 16, 56, 56, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape56_3(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 16, 56, 56, 16), 'float16', (1, 1), (2, 2)):
        tik_instance, res = shape56_4(tik_instance, input_x, res, input_shape, shape_info)

    return tik_instance, res


def height28_width28(tik_instance, input_x, res, input_shape, shape_info):
    """height28_width28"""
    if input_shape == ((32, 8, 28, 28, 16), 'float16', (3, 3), (1, 1)):
        tik_instance, res = shape28_0(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 16, 28, 28, 16), 'float16', (3, 3), (2, 2)):
        tik_instance, res = shape28_1(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 32, 28, 28, 16), 'float16', (1, 1), (2, 2)):
        tik_instance, res = shape28_2(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 8, 28, 28, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape28_3(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 32, 28, 28, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape28_4(tik_instance, input_x, res, input_shape, shape_info)

    return tik_instance, res


def height14_width14(tik_instance, input_x, res, input_shape, shape_info):
    """height14_width14"""
    if input_shape == ((32, 16, 14, 14, 16), 'float16', (3, 3), (1, 1)):
        tik_instance, res = shape14_0(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 32, 14, 14, 16), 'float16', (3, 3), (2, 2)):
        tik_instance, res = shape14_1(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 64, 14, 14, 16), 'float16', (1, 1), (2, 2)):
        tik_instance, res = shape14_2(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 64, 14, 14, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape14_3(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 16, 14, 14, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape14_4(tik_instance, input_x, res, input_shape, shape_info)

    return tik_instance, res


def height7_width7(tik_instance, input_x, res, input_shape, shape_info):
    """height7_width7"""
    if input_shape == ((32, 32, 7, 7, 16), 'float16', (3, 3), (1, 1)):
        tik_instance, res = shape7_0(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 128, 7, 7, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape7_1(tik_instance, input_x, res, input_shape, shape_info)

    if input_shape == ((32, 32, 7, 7, 16), 'float16', (1, 1), (1, 1)):
        tik_instance, res = shape7_2(tik_instance, input_x, res, input_shape, shape_info)

    return tik_instance, res


@op_info_register(cus_img2col_info)
def cus_img2col(input_x, output, ksizes, strides, dilates, mode, kernel_name="img2col"):
    """CusImg2Col"""
    input_x_shape = input_x.get("shape")
    input_x_dtype = input_x.get("dtype")
    n_shape, c1_shape, h_shape, w_shape, c0_shape = input_x_shape
    c_shape = c1_shape * c0_shape
    _, filter_h, filter_w, _ = ksizes
    _, stride_h, stride_w, _ = strides
    _, dilation_filter_h, dilation_filter_w, _ = dilates
    shape_info = stride_w, stride_h, filter_w, filter_h, dilation_filter_w, dilation_filter_h

    input_shape = (tuple(input_x_shape), input_x_dtype, (filter_h, filter_w), (stride_h, stride_w))
    supported_shape = [((32, 32, 14, 14, 16), 'float16', (3, 3), (2, 2)),
                       ((32, 1, 224, 224, 16), 'float16', (7, 7), (2, 2)),
                       ((32, 4, 56, 56, 16), 'float16', (3, 3), (1, 1)),
                       ((32, 8, 56, 56, 16), 'float16', (3, 3), (2, 2)),
                       ((32, 8, 28, 28, 16), 'float16', (3, 3), (1, 1)),
                       ((32, 16, 28, 28, 16), 'float16', (3, 3), (2, 2)),
                       ((32, 16, 14, 14, 16), 'float16', (3, 3), (1, 1)),
                       ((32, 32, 7, 7, 16), 'float16', (3, 3), (1, 1)),
                       ((32, 64, 14, 14, 16), 'float16', (1, 1), (1, 1)),
                       ((32, 32, 7, 7, 16), 'float16', (1, 1), (1, 1)),
                       ((32, 4, 56, 56, 16), 'float16', (1, 1), (1, 1)),
                       ((32, 64, 14, 14, 16), 'float16', (1, 1), (2, 2)),
                       ((32, 128, 7, 7, 16), 'float16', (1, 1), (1, 1)),
                       ((32, 32, 28, 28, 16), 'float16', (1, 1), (2, 2)),
                       ((32, 16, 56, 56, 16), 'float16', (1, 1), (2, 2)),
                       ((32, 8, 28, 28, 16), 'float16', (1, 1), (1, 1)),
                       ((32, 32, 28, 28, 16), 'float16', (1, 1), (1, 1)),
                       ((32, 16, 14, 14, 16), 'float16', (1, 1), (1, 1)),
                       ((32, 16, 56, 56, 16), 'float16', (1, 1), (1, 1))]

    if input_shape not in supported_shape:
        raise RuntimeError("input_shape %s is not supported" % str(input_shape))

    output_tmp = [n_shape * int(h_shape // stride_h) * int(w_shape // stride_w), filter_h * filter_w * c_shape]
    output_shape = [output_tmp[1] // 16, output_tmp[0] // 16, 16, 16]
    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))

    input_x = tik_instance.Tensor("float16", input_x_shape, name="input_x", scope=tik.scope_gm)
    res = tik_instance.Tensor("float16", output_shape, name="res", scope=tik.scope_gm)

    if tuple([h_shape, w_shape]) == (224, 224):
        tik_instance, res = height224_width224(tik_instance, input_x, res, input_shape, shape_info)

    if tuple([h_shape, w_shape]) == (56, 56):
        tik_instance, res = height56_width56(tik_instance, input_x, res, input_shape, shape_info)

    if tuple([h_shape, w_shape]) == (28, 28):
        tik_instance, res = height28_width28(tik_instance, input_x, res, input_shape, shape_info)

    if tuple([h_shape, w_shape]) == (14, 14):
        tik_instance, res = height14_width14(tik_instance, input_x, res, input_shape, shape_info)

    if tuple([h_shape, w_shape]) == (7, 7):
        tik_instance, res = height7_width7(tik_instance, input_x, res, input_shape, shape_info)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance
