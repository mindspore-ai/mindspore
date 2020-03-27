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
"""Conv2D_transpose of stride=2, kernel=2*2 schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import functools

import tvm
from tvm import autotvm
import tvm.contrib.nnpack

from topi.generic import schedule_conv2d_nchw
from topi.util import traverse_inline, get_const_tuple
from topi.nn import conv2d


@autotvm.register_topi_compute(conv2d, "arm_cpu", ["deconv"])
def conv2d_arm_cpu_deconv(cfg, data, kernel, out_dtype):
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

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    args = _gen_cfg_deconv(cfg, data, kernel, num_tile=2)
    return _conv_spatial_pack_deconv(
        args, data, kernel, out_dtype
    )


@autotvm.register_topi_schedule(schedule_conv2d_nchw, "arm_cpu", ["deconv"])
def schedule_conv2d_nchw_arm_cpu_deconv(cfg, outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    s = _conv_schedule_deconv(cfg, outs)
    return s


def _gen_cfg_deconv(cfg, data, kernel, num_tile):
    """generation config from input args"""
    if len(kernel.shape) == 4:
        co_, _, _, _ = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        co_, _, _, _, vc_ = get_const_tuple(kernel.shape)
        co_ = co_ * vc_

    if len(data.shape) == 4:
        _, ci_, ih_, iw_ = get_const_tuple(data.shape)
        c4 = 4
        ci_ = ci_ // 4
    else:
        _, ci_, ih_, iw_, c4 = get_const_tuple(data.shape)

    oh_ = ih_ * 2
    ow_ = iw_ * 2

    co, oh, ow = cfg.axis(co_), cfg.axis(oh_), cfg.axis(ow_)
    ci, ki = cfg.reduce_axis(ci_), cfg.reduce_axis(c4)

    if num_tile == 2:  # for arm cpu
        candidate_vc = [[co_ // c4, c4]]
        co, vc = cfg.define_split(
            "tile_co", co, num_outputs=2, policy="candidate", candidate=candidate_vc
        )
        candidate_vw = []
        for iv in range(4, ow_ + 1):  # [4, 6, 8, 12, 16, 24, 32, 40]:
            if iv % 4 == 0 and (ow_ % iv == 0):
                candidate_vw.append([ow_ // iv, iv])
        ow, vw = cfg.define_split(
            "tile_ow", ow, num_outputs=2, policy="candidate", candidate=candidate_vw
        )
        candidate_vh = [[1, 2]]
        oh, vh = cfg.define_split(
            "tile_oh", oh, num_outputs=2, policy="candidate", candidate=candidate_vh
        )
    elif num_tile == 3:  # for mali gpu
        co, _, vc = cfg.define_split("tile_co", co, num_outputs=3)
        oh, _, vh = cfg.define_split("tile_oh", oh, num_outputs=3)
        ow, _, vw = cfg.define_split("tile_ow", ow, num_outputs=3)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_annotate("ann_reduce", [ci, ki], policy="try_unroll")
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy="try_unroll_vec")

    vc_ = cfg["tile_co"].size[-1]
    vh_ = cfg["tile_oh"].size[-1]
    vw_ = cfg["tile_ow"].size[-1]
    is_var = False
    return (is_var, vh_, vw_, vc_)


def _conv_spatial_pack_deconv(args, data, kernel, out_dtype):
    """conv2d_arm_cpu_deconv inner implement"""
    is_var, vh_, vw_, vc_ = args
    # create workload according to raw arguments
    out_dtype = out_dtype or data.dtype
    if len(data.shape) == 4:
        n_, ci_, ih_, iw_ = data.shape if is_var else get_const_tuple(data.shape)
        c4 = 4
        ci_ = ci_ // c4
    else:
        n_, ci_, ih_, iw_, c4 = data.shape if is_var else get_const_tuple(data.shape)

    if len(kernel.shape) == 4:
        pre_packed = False
        _, co_, kh_, kw_ = kernel.shape if is_var else get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        _, co_, kh_, kw_, vc_ = kernel.shape if is_var else get_const_tuple(kernel.shape)
        co_ = co_ * c4

    oh_ = ih_ * 2
    ow_ = iw_ * 2
    ow_div = ow_ // vw_
    oh_div = oh_ // vh_
    kvshape = (co_ // vc_, kh_, kw_, ci_, c4, c4)
    ovshape = (n_, co_ // vc_, oh_div, ow_div, vh_, vw_, c4)

    dvshape = (n_, ih_ // (vh_ // 2), iw_ // (vw_ // 2), vh_ // 2, ci_, vw_ // 2, c4)
    if len(data.shape) == 4:
        data_vec = tvm.compute(
            dvshape,
            lambda n, h, w, vh, ci, vw, ki: data[n][ci * c4 + ki][h * vh_ // 2 + vh][
                w * vw_ // 2 + vw
            ],
            name="data_vec",
        )
    else:
        data_vec = tvm.compute(
            dvshape,
            lambda n, h, w, vh, ci, vw, ki: data[n][ci][h * vh_ // 2 + vh][
                w * vw_ // 2 + vw
            ][ki],
            name="data_vec",
        )

    if pre_packed:
        kernel_vec = kernel
    else:
        kernel_vec = tvm.compute(
            kvshape,
            lambda co, kh, kw, ci, ki, vc: kernel[ci * c4 + ki][co * vc_ + vc][kh][kw],
            name="kernel_vec",
        )

    ci = tvm.reduce_axis((0, ci_), name="ci")
    ki = tvm.reduce_axis((0, c4), name="ki")

    type_map = {
        "int8": "int32",
        "uint8": "uint32",
        "float32": "float32",
        "float16": "float16",
    }
    acum_dtype = type_map[data.dtype]
    attrs = {
        "SH": 2,
        "SW": 2,
        "PH": 0,
        "PW": 0,
        "DILA_H": 1,
        "DILA_W": 1,
        "VH": vh_,
        "VW": vw_,
        "VC": vc_,
        "ACUM_DTYPE": acum_dtype,
    }

    conv = tvm.compute(
        ovshape,
        lambda n, co, h, w, vh, vw, vc: tvm.sum(
            data_vec[n, h, w, vh // 2, ci, vw // 2, ki].astype(out_dtype)
            * kernel_vec[co, (h * vh_ + vh) % 2, (w * vw_ + vw) % 2, ci, ki, vc].astype(
                out_dtype
            ),
            axis=[ci, ki],
        ),
        name="conv",
        attrs=attrs,
    )
    if len(data.shape) == 4:
        osshape = (n_, co_, oh_, ow_div * vw_)
        output = tvm.compute(
            osshape,
            lambda n, co, h, w: conv[n][co // c4][h][w // vw_][w % vw_][co % c4],
            name="output_unpack",
            tag="deconv_conv2d_output",
        )
    else:
        osshape = (n_, co_ // c4, oh_, ow_div * vw_, c4)
        output = tvm.compute(
            osshape,
            lambda n, co, h, w, vc: conv[n][co][h // vh_][w // vw_][h % vh_][w % vw_][vc],
            name="output_unpack",
            tag="deconv_conv2d_output",
        )

    return output


def intrin_deconv(args):
    """deconv inner implement"""
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
    if DILA_H != 1 or dila_w != 1:
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


def _schedule_deconv(cfg, s, data_vec, kernel_vec, conv, output, last):
    """schedule implementation"""
    is_tune = bool(isinstance(cfg, (tvm.autotvm.ConfigEntity, tvm.autotvm.ConfigSpace)))
    if is_tune:
        vh_ = cfg["tile_oh"].size[-1]
        vw_ = cfg["tile_ow"].size[-1]
        vc_ = cfg["tile_co"].size[-1]
        cfg = {
            "ci_": tvm.var("ci_"),
            "VH": vh_,
            "VW": vw_,
            "VC": vc_,
            "tile_oh": vh_,
            "tile_ow": vw_,
            "tile_co": vc_,
            "tile_ci": 4,
            "ann_reduce": cfg["ann_reduce"].anns,
            "ann_spatial": cfg["ann_spatial"].anns,
        }  # ,'reorder_0':cfg['reorder_0'].perm}
    else:
        pass
    n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    ci, ki = s[conv].op.reduce_axis
    s[conv].reorder(n, co, oh, ow, ci, vw, ki, vc)
    if cfg["ann_reduce"][0] == "unroll":
        s[conv].unroll(ci)
    elif cfg["ann_reduce"][0] == "vec":
        s[conv].vectorize(ci)
    if cfg["ann_reduce"][1] == "unroll":
        s[conv].unroll(ki)
    elif cfg["ann_reduce"][1] == "vec":
        s[conv].vectorize(ki)
    if cfg["ann_spatial"][0] == "vec":
        s[conv].vectorize(vh)
    elif cfg["ann_spatial"][0] == "unroll":
        s[conv].unroll(vh)
    if cfg["ann_spatial"][1] == "vec":
        s[conv].vectorize(vw)
    elif cfg["ann_spatial"][1] == "unroll":
        s[conv].unroll(vw)
    if cfg["ann_spatial"][2] == "vec":
        s[conv].vectorize(vc)
    elif cfg["ann_spatial"][2] == "unroll":
        s[conv].unroll(vc)

    # schedule conv
    attrs = conv.op.attrs
    vh_, vw_, vc_ = (attrs["VH"].value, attrs["VW"].value, attrs["VC"].value)

    # schedule fusion
    if len(s[last].op.axis) == 4:
        n, co, h, w = s[last].op.axis
        co, vc = s[last].split(co, vc_)
        ow, vw = s[last].split(w, vw_)
        oh, vh = s[last].split(h, vh_)
        s[last].reorder(n, co, oh, ow, vh, vw, vc)
    else:
        n, co, h, w, vc = s[last].op.axis
        oh, vh = s[last].split(h, vh_)
        ow, vw = s[last].split(w, vw_)
        s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output and isinstance(output.op, tvm.tensor.ComputeOp):
        s[output].compute_inline()
        if cfg["ann_spatial"][0] == "vec":
            s[last].vectorize(vh)
        elif cfg["ann_spatial"][0] == "unroll":
            s[last].unroll(vh)
        if cfg["ann_spatial"][1] == "vec":
            s[last].vectorize(vw)
        elif cfg["ann_spatial"][1] == "unroll":
            s[last].unroll(vw)
        if cfg["ann_spatial"][2] == "vec":
            s[last].vectorize(vc)
        elif cfg["ann_spatial"][2] == "unroll":
            s[last].unroll(vc)

    s[conv].compute_at(s[last], ow)

    # mark parallel
    s[last].parallel(co)

    if data_vec.op.name == "data_vec_undilated":
        _, h, _, _, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, h, _, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    co, _, _, _, _, vc = s[kernel_vec].op.axis
    s[kernel_vec].parallel(co)
    if cfg["ann_spatial"][2] == "vec":
        s[kernel_vec].vectorize(vc)
    elif cfg["ann_spatial"][2] == "unroll":
        s[kernel_vec].unroll(vc)
    return s


def _conv_schedule_deconv(cfg, outs):
    """schedule_conv2d_nchw_arm_cpu_deconv inner implementation"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "deconv_conv2d_output" in op.tag:
            # schedule conv2d
            output = op.output(0)
            conv = op.input_tensors[0]

            sidx = 0
            if conv.op.input_tensors[0].name == "attr":
                sidx = 1
            data_vec = conv.op.input_tensors[sidx]

            kernel_vec = conv.op.input_tensors[sidx + 1]
            if kernel_vec.op.name == "kernel_vec":
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if (isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag):
                s[kernel].compute_inline()

            _schedule_deconv(cfg, s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s
