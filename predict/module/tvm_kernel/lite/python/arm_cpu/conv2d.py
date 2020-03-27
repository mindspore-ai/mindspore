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
"""Conv2D schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import functools

import tvm
from tvm import autotvm
import tvm.contrib.nnpack

from topi.generic import schedule_conv2d_nchw
from topi.util import traverse_inline, get_const_tuple
from topi.nn import pad, conv2d
from topi.nn.util import get_const_int, get_pad_tuple


@autotvm.register_topi_compute(conv2d, "arm_cpu", ["asm"])
def conv2d_arm_cpu(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TOPI compute callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        pre-packed 5-D with shape [num_filter_chunk, in_channel, filter_height,
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
    args = _gen_cfg(cfg, data, kernel, strides, padding, dilation, num_tile=2)
    return _conv_spatial_pack_asm(
        args, data, kernel, strides, padding, dilation, out_dtype
    )


@autotvm.register_topi_schedule(schedule_conv2d_nchw, "arm_cpu", ["asm"])
def schedule_conv2d_nchw_arm_cpu(outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    s = _conv_schedule_asm(outs)
    return s


def _gen_cfg(cfg, data, kernel, strides, padding, dilation, num_tile):
    """_gen_cfg"""
    if len(kernel.shape) == 4:
        co_, _, kh_, kw_ = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        co_, _, kh_, kw_, vc_ = get_const_tuple(kernel.shape)
        co_ = co_ * vc_

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    n_, ci_, ih_, iw_ = get_const_tuple(data.shape)

    dilated_kernel_h = (kh_ - 1) * dilation_h + 1
    dilated_kernel_w = (kw_ - 1) * dilation_w + 1
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    hstr, wstr = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh_ = (ih_ + pad_top + pad_bottom - dilated_kernel_h) // hstr + 1
    ow_ = (iw_ + pad_left + pad_right - dilated_kernel_w) // wstr + 1

    n, co, oh, ow = cfg.axis(n_), cfg.axis(co_), cfg.axis(oh_), cfg.axis(ow_)
    ci, kh, kw = cfg.reduce_axis(ci_), cfg.reduce_axis(kh_), cfg.reduce_axis(kw_)

    if num_tile == 2:  # for arm cpu
        candidate_vc = []
        for iv in range(3, co_):
            if co_ % iv == 0:
                candidate_vc.append([co_ // iv, iv])
        candidate_vc.append([1, co_])
        co, vc = cfg.define_split(
            "tile_co", co, num_outputs=2, policy="candidate", candidate=candidate_vc
        )
        oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
        ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)
    elif num_tile == 3:  # for mali gpu
        co, _, vc = cfg.define_split("tile_co", co, num_outputs=3)
        oh, _, vh = cfg.define_split("tile_oh", oh, num_outputs=3)
        ow, _, vw = cfg.define_split("tile_ow", ow, num_outputs=3)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_reorder(
        "reorder_0",
        [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
        policy="candidate",
        candidate=[[n, co, oh, ow, ci, kh, kw, vh, vw, vc],],
    )

    vc_ = cfg["tile_co"].size[-1]
    vh_ = cfg["tile_oh"].size[-1]
    vw_ = cfg["tile_ow"].size[-1]
    is_var = False
    return (is_var, vh_, vw_, vc_)

def _conv_spatial_pack_asm(args, data, kernel, strides, padding,
                           dilation, out_dtype):
    """_conv_spatial_pack_asm"""
    is_var, vh_, vw_, vc_ = args

    # create workload according to raw arguments
    out_dtype = out_dtype or data.dtype
    n_, ci_, ih_, iw_ = data.shape if is_var else get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        pre_packed = False
        co_, _, kh_, kw_ = kernel.shape if is_var else get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        co_, _, kh_, kw_, vc_ = kernel.shape if is_var else get_const_tuple(kernel.shape)
        co_ = co_ * vc_

    dilated_kernel_h = (kh_ - 1) * dilation_h + 1
    dilated_kernel_w = (kw_ - 1) * dilation_w + 1
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    hstr, wstr = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh_ = (ih_ + pad_top + pad_bottom - dilated_kernel_h) // hstr + 1
    ow_ = (iw_ + pad_left + pad_right - dilated_kernel_w) // wstr + 1
    data_pad = pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right])

    oh_div = oh_ // vh_
    ow_div = ow_ // vw_
    kvshape = (co_ // vc_, ci_, kh_, kw_, vc_)
    ovshape = (n_, co_ // vc_, oh_div, ow_div, vh_, vw_, vc_)
    oshape = (n_, co_, oh_div * vh_, ow_div * vw_)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (n_, oh_ // vh_, ow_ // vw_, kh_, kw_, vh_, vw_, ci_)
        data_vec = tvm.compute(
            dvshape,
            lambda n, h, w, kh, kw, vh, vw, ci: data_pad[n][ci][
                (h * vh_ + vh) * hstr + kh * dilation_h
            ][(w * vw_ + vw) * wstr + kw * dilation_w],
            name="data_vec_undilated",
        )
    else:
        dvshape = (
            n_,
            oh_ // vh_,
            ow_ // vw_,
            (vh_ - 1) * hstr + kh_,
            (vw_ - 1) * wstr + kw_,
            ci_,
        )
        data_vec = tvm.compute(
            dvshape,
            lambda n, h, w, vh, vw, ci: data_pad[n][ci][h * vh_ * hstr + vh][
                w * vw_ * wstr + vw
            ],
            name="data_vec",
        )

    if pre_packed:
        kernel_vec = kernel
    else:
        kernel_vec = tvm.compute(
            kvshape,
            lambda co, ci, kh, kw, vc: kernel[co * vc_ + vc][ci][kh][kw],
            name="kernel_vec",
        )

    ci = tvm.reduce_axis((0, ci_), name="ci")
    kh = tvm.reduce_axis((0, kh_), name="kh")
    kw = tvm.reduce_axis((0, kw_), name="kw")

    # asm begin----
    type_map = {
        "int8": "int32",
        "uint8": "uint32",
        "float32": "float32",
        "float16": "float16",
    }
    acum_dtype = type_map[data.dtype]
    attrs = {
        "SH": hstr,
        "SW": wstr,
        "PH": pad_top,
        "PW": pad_left,
        "DILA_H": dilation_h,
        "DILA_W": dilation_w,
        "VH": vh_,
        "VW": vw_,
        "VC": vc_,
        "ACUM_DTYPE": acum_dtype,
    }
    # asm end----

    if dilation_h != 1 or dilation_w != 1:
        conv = tvm.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: tvm.sum(
                data_vec[n, h, w, kh, kw, vh, vw, ci].astype(out_dtype)
                * kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                axis=[ci, kh, kw],
            ),
            name="conv",
            attrs=attrs,
        )
    else:
        conv = tvm.compute(
            ovshape,
            lambda n, co, h, w, vh, vw, vc: tvm.sum(
                data_vec[n, h, w, vh * hstr + kh, vw * wstr + kw, ci].astype(out_dtype)
                * kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                axis=[ci, kh, kw],
            ),
            name="conv",
            attrs=attrs,
        )

    output = tvm.compute(
        oshape,
        lambda n, co, h, w: conv[n][co // vc_][h // vh_][w // vw_][h % vh_][w % vw_][
            co % vc_
        ],
        name="output_unpack",
        tag="asm_conv2d_output",
    )

    return output


def intrin_conv(args):
    """intrin_conv"""
    (
        ci_,
        vh_,
        vw_,
        vc_,
        kh_,
        kw_,
        sh_,
        sw_,
        dila_h,
        dila_w,
        dtype,
        acum_dtype,
        opname,
        core_id,
    ) = args
    hstr, wstr = sh_, sw_
    ci_ = tvm.var("ci_") if ci_ is None else ci_
    kvshape = (ci_, kh_, kw_, vc_)
    ovshape = (vh_, vw_, vc_)
    if dila_h != 1 or dila_w != 1:
        dvshape = (kh_, kw_, vh_, vw_, ci_)
    else:
        dvshape = ((vh_ - 1) * hstr + kh_, (vw_ - 1) * wstr + kw_, ci_)

    data_vec = tvm.placeholder(dvshape, name="a", dtype=dtype)
    kernel_vec = tvm.placeholder(kvshape, name="b", dtype=dtype)
    ci = tvm.reduce_axis((0, ci_), name="ci")
    kh = tvm.reduce_axis((0, kh_), name="kh")
    kw = tvm.reduce_axis((0, kw_), name="kw")
    if dila_h != 1 or dila_w != 1:
        conv = tvm.compute(
            ovshape,
            lambda vh, vw, vc: tvm.sum(
                data_vec[kh, kw, vh, vw, ci].astype(acum_dtype)
                * kernel_vec[ci, kh, kw, vc].astype(acum_dtype),
                axis=[ci, kh, kw],
            ),
            name="conv",
        )
    else:
        conv = tvm.compute(
            ovshape,
            lambda vh, vw, vc: tvm.sum(
                data_vec[vh * hstr + kh, vw * wstr + kw, ci].astype(acum_dtype)
                * kernel_vec[ci, kh, kw, vc].astype(acum_dtype),
                axis=[ci, kh, kw],
            ),
            name="conv",
        )

    stride_a = [
        functools.reduce(lambda x, y: x * y, dvshape[i + 1: len(dvshape)])
        for i in range(0, len(dvshape) - 1)
    ]
    stride_a.append(1)
    stride_b = [
        functools.reduce(lambda x, y: x * y, kvshape[i + 1: len(kvshape)])
        for i in range(0, len(kvshape) - 1)
    ]
    stride_b.append(1)
    stride_c = [
        functools.reduce(lambda x, y: x * y, ovshape[i + 1: len(ovshape)])
        for i in range(0, len(ovshape) - 1)
    ]
    stride_c.append(1)

    a_buffer = tvm.decl_buffer(
        data_vec.shape, data_vec.dtype, name="A", offset_factor=1, strides=stride_a
    )
    b_buffer = tvm.decl_buffer(
        kernel_vec.shape, kernel_vec.dtype, name="B", offset_factor=1, strides=stride_b
    )
    c_buffer = tvm.decl_buffer(
        conv.shape, conv.dtype, name="C", offset_factor=1, strides=stride_c
    )

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.ir_builder.create()
            ib.emit(
                tvm.call_extern(
                    "int32",
                    opname,
                    cc.access_ptr("w"),
                    aa.access_ptr("r"),
                    bb.access_ptr("r"),
                    ci_,
                    vh_,
                    vw_,
                    vc_,
                    kh_,
                    sh_,
                    core_id,
                )
            )
            return ib.get()

        return _body()

    return tvm.decl_tensor_intrin(
        conv.op, intrin_func, binds={data_vec: a_buffer, kernel_vec: b_buffer, conv: c_buffer}
    )


def _schedule_asm(s, data_vec, kernel_vec, conv, output, last):
    """schedule implementation"""
    n, co, oh, ow, vh, vw, vc = s[conv].op.axis

    axis_extent = []
    for i in (vh, vw, vc):
        axis_extent.append(get_const_int(i.dom.extent))
    reduce_extent = []
    for i in s[conv].op.reduce_axis[1:]:
        reduce_extent.append(get_const_int(i.dom.extent))
    vh_, vw_, vc_ = axis_extent

    # schedule fusion
    n, co, h, w = s[last].op.axis
    co, vc = s[last].split(co, vc_)
    oh, vh = s[last].split(h, vh_)
    ow, vw = s[last].split(w, vw_)
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()

    s[conv].compute_at(s[last], ow)

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
            # kernel packing will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, "debug_skip_region")
        else:
            s[kernel_vec].parallel(co)
    elif kernel_vec.op.name == "kernel_vec_conv2d_transpose":  # for conv2d transpose
        co, _, _, _, _ = s[kernel_vec].op.axis
        s[kernel_vec].parallel(co)

    return s


def _conv_schedule_asm(outs):
    """_conv_schedule_asm"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "asm_conv2d_output" in op.tag:
            # schedule conv2d
            output = op.output(0)
            conv = op.input_tensors[0]

            sidx = 0
            if conv.op.input_tensors[0].name == "attr":
                sidx = 1
            data_vec = conv.op.input_tensors[sidx]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[sidx + 1]
            if kernel_vec.op.name == "kernel_vec":
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if (isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag):
                s[kernel].compute_inline()

            if conv.op.input_tensors[0].name == "attr":
                _schedule_asm(s, data_vec, kernel_vec, conv, output, outs[0])
            else:
                _schedule_asm(s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s
