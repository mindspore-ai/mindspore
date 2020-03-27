# Copyright 2019 Huawei Technologies Co., Ltd
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
"""Depthwise convolution schedule for ARM CPU"""

import tvm
from tvm import autotvm

from topi.generic import schedule_depthwise_conv2d_nchw
from topi.nn import depthwise_conv2d_nchw, pad
from topi.util import traverse_inline, get_const_tuple
from topi.nn.util import get_pad_tuple

# register customized schedule for arm cpu.
@autotvm.register_topi_schedule(
    schedule_depthwise_conv2d_nchw, ["arm_cpu", "cpu"], ["custom"]
)
def schedule_depthwise_conv2d_nchw_arm(cfg, outs):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    s = _depthwise_schedule_spatial_pack(cfg, outs)
    return s


@autotvm.register_topi_compute(depthwise_conv2d_nchw, ["arm_cpu", "cpu"], ["custom"])
def depthwise_conv2d_arm_cpu(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TOPI compute callback for depthwise_conv2d nchw

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        4-D with shape [num_filter, multiplier, filter_height, filter_width] or
        pre-packed 5-D with shape [num_filter_chunk, multiplier, filter_height,
        filter_width, num_filter_block]

    strides : list of two ints
        [stride_height, stride_width]

    padding : list of two ints
        [pad_height, pad_width]

    dilation : list of two ints
        [dilation_height, dilation_width]

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    return _depthwise_spatial_pack(
        cfg, data, kernel, strides, padding, dilation, out_dtype
    )


def _depthwise_spatial_pack(args, data, kernel, strides, padding, dilation, out_dtype):
    """depthwise_conv2d_arm_cpu's inner implement"""
    is_var, u_vh, u_vw, u_vc = args
    out_dtype = out_dtype or data.dtype

    u_n, u_c, ih, iw = data.shape if is_var else get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        pre_packed = False
        u_c, um, ukh, ukw = kernel.shape if is_var else get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        u_c, um, ukh, ukw, u_vc = kernel.shape if is_var else get_const_tuple(kernel.shape)
        u_c = u_c * u_vc

    dilated_kernel_h = (ukh - 1) * dilation_h + 1
    dilated_kernel_w = (ukw - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    hstr, wstr = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    u_oh = (ih + pad_top + pad_down - dilated_kernel_h) // hstr + 1
    u_ow = (iw + pad_left + pad_right - dilated_kernel_w) // wstr + 1
    # pack data
    hpad = pad_top + pad_down
    wpad = pad_left + pad_right
    dopad = hpad != 0 or wpad != 0
    if dopad:
        data_pad = pad(
            data,
            (0, 0, pad_top, pad_left),
            (0, 0, pad_down, pad_right),
            name="data_pad",
        )
    else:
        data_pad = data

    oh_div = u_oh // u_vh
    ow_div = u_ow // u_vw
    kvshape = (u_c // u_vc, um, ukh, ukw, u_vc)
    ovshape = (u_n, u_c * um // u_vc, oh_div, u_ow // u_vw, u_vh, u_vw, u_vc)
    oshape = (u_n, u_c * um, oh_div * u_vh, ow_div * u_vw)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (u_n, oh_div, ow_div, u_c, ukh, ukw, u_vh, u_vw)
        data_vec = tvm.compute(
            dvshape,
            lambda n, h, w, c, kh, kw, vh, vw: data_pad[n][c][
                (h * u_vh + vh) * hstr + kh * dilation_h
            ][(w * u_vw + vw) * wstr + kw * dilation_w],
            name="data_vec_undilated",
        )
    else:
        dvshape = (u_n, oh_div, ow_div, u_c, u_vh * hstr + ukh - 1, u_vw * wstr + ukw - 1)
        data_vec = tvm.compute(
            dvshape,
            lambda n, h, w, c, vh, vw: data_pad[n][c][h * u_vh * hstr + vh][
                w * u_vw * wstr + vw
            ],
            name="data_vec",
        )

    if pre_packed:
        kernel_vec = kernel
    else:
        kernel_vec = tvm.compute(
            kvshape,
            lambda co, m, kh, kw, vc: kernel[co * u_vc + vc][m][kh][kw],
            name="kernel_vec",
        )

    kh = tvm.reduce_axis((0, ukh), name="kh")
    kw = tvm.reduce_axis((0, ukw), name="kw")

    if dilation_h != 1 or dilation_w != 1:
        conv = tvm.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: tvm.sum(
                data_vec[n, h, w, (co * u_vc + vc) // um, kh, kw, vh, vw].astype(out_dtype)
                * kernel_vec[co // um, co % um, kh, kw, vc].astype(out_dtype),
                axis=[kh, kw],
            ),
            name="depthwise_conv",
        )
    else:
        conv = tvm.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: tvm.sum(
                data_vec[
                    n, h, w, (co * u_vc + vc) // um, vh * hstr + kh, vw * wstr + kw
                ].astype(out_dtype)
                * kernel_vec[co // um, co % um, kh, kw, vc].astype(out_dtype),
                axis=[kh, kw],
            ),
            name="depthwise_conv",
        )

    output = tvm.compute(
        oshape,
        lambda n, co, h, w: conv[n][co // u_vc][h // u_vh][w // u_vw][h % u_vh][w % u_vw][
            co % u_vc
        ],
        name="output_unpack",
        tag="spatial_depthwise_conv_nchw_output",
    )
    return output


def _schedule_spatial_pack(cfg, s, data_vec, kernel_vec, conv, output, last):
    """schedule implementation"""
    u_vc = cfg["tile_co"].size[-1] if not isinstance(cfg, dict) else cfg["VC"]
    u_vh = cfg["tile_oh"].size[-1] if not isinstance(cfg, dict) else cfg["VH"]
    u_vw = cfg["tile_ow"].size[-1] if not isinstance(cfg, dict) else cfg["VW"]

    n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    kh, kw = s[conv].op.reduce_axis

    if data_vec.op.name == "data_vec_undilated":
        _, _, dv_ow, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, _, dv_ow, _, _, _ = s[data_vec].op.axis

    data_pad = data_vec.op.input_tensors[0]

    if isinstance(data_pad.op, tvm.tensor.ComputeOp):
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
    s[data_pad].compute_at(s[data_vec], dv_ow)

    s[data_vec].vectorize(list(s[data_vec].op.axis)[-1])
    s[data_vec].compute_at(s[conv], ow)

    # schedule conv
    s[conv].reorder(n, co, oh, ow, kh, kw, vh, vw, vc)
    s[conv].unroll(kh)
    s[conv].unroll(vh)
    s[conv].vectorize(vw)
    s[conv].unroll(vc)
    s[conv].parallel(co)

    n, co, h, w = s[last].op.axis
    co, vc = s[last].split(co, u_vc)
    oh, vh = s[last].split(h, u_vh)
    ow, vw = s[last].split(w, u_vw)
    if last != output:
        s[output].compute_inline()
        s[last].vectorize(vw)
        s[last].unroll(vc)
    else:
        s[last].vectorize(vw)
    s[conv].compute_at(s[last], oh)

    # mark parallel
    s[last].parallel(co)

    if data_vec.op.name == "data_vec_undilated":
        _, h, _, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, h, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    if kernel_vec.op.name == "kernel_vec":
        co, _, _, _, _ = s[kernel_vec].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compliation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, "debug_skip_region")
        else:
            s[kernel_vec].parallel(co)

    return s


def _depthwise_schedule_spatial_pack(cfg, outs):
    """schedule_depthwise_conv2d_nchw_arm's inner implement"""
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "spatial_depthwise_conv_nchw_output":
            output = op.output(0)
            conv = op.input_tensors[0]
            data_vec = conv.op.input_tensors[0]
            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == "kernel_vec":
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            _schedule_spatial_pack(cfg, s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s
