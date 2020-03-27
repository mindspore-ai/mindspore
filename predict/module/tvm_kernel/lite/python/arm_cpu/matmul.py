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
from topi.util import traverse_inline
from topi.nn import conv2d


@autotvm.register_topi_compute(conv2d, "arm_cpu", ["matmul"])
def matmul_arm_cpu(cfg, a_, b_, layout, out_dtype):
    """TOPI compute callback for

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    a_ : tvm.Tensor
        2-D with shape [M, k_]

    b_ : tvm.Tensor
        2-D with shape [k_, N]

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    args = _gen_cfg(cfg, a_, b_)
    return _matmul_spatial_pack_asm(args, a_, b_, layout, out_dtype)


@autotvm.register_topi_schedule(schedule_conv2d_nchw, "arm_cpu", ["matmul"])
def schedule_matmul_arm_cpu(cfg, outs):
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
    s = _matmul_schedule_asm(cfg, outs)
    return s


def _gen_cfg(cfg, a_, b_):
    """get best loginfo from cfg"""
    if len(a_.shape) == 2:
        w_, ci_ = get_const_tuple(a_.shape)
        h_ = 1
    elif len(a_.shape) == 3:
        _, ci_, w_ = get_const_tuple(a_.shape)
        h_ = 1
    elif len(a_.shape) == 4:
        _, ci_, h_, w_ = get_const_tuple(a_.shape)
    else:
        raise ValueError("not support shape: " + a_.shape)

    co_, k_ = get_const_tuple(b_.shape)

    oh, ow = cfg.axis(h_), cfg.axis(w_)
    co = cfg.axis(co_)
    k = cfg.reduce_axis(k_)

    oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2)
    ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2)
    oc, vc = cfg.define_split("tile_co", co, num_outputs=2)

    cfg.define_reorder(
        "reorder_0",
        [oc, oh, ow, k, vh, vw, vc],
        policy="candidate",
        candidate=[[oc, oh, ow, k, vh, vw, vc],],
    )

    vh_ = cfg["tile_oh"].size[-1]
    vw_ = cfg["tile_ow"].size[-1]
    vc_ = cfg["tile_co"].size[-1]
    is_var = False
    is_transpose = False
    return (is_var, is_transpose, ci_, vh_, vw_, vc_)


def _matmul_spatial_pack_asm(args, a_, b_, layout, out_dtype):
    """matmul_spatial_pack_asm's inner interace"""
    is_var, is_transpose, ci_, vh_, vw_, vc_ = args

    # create workload according to raw arguments
    out_dtype = out_dtype or a_.dtype
    if layout == "NCHW":
        batch, k_, h_, w_ = a_.shape if is_var else get_const_tuple(a_.shape)
        n_, _ = b_.shape if is_var else get_const_tuple(b_.shape)
    elif layout == "NCH":
        batch, k_, h_ = a_.shape if is_var else get_const_tuple(a_.shape)
        n_, _ = b_.shape if is_var else get_const_tuple(b_.shape)
        w_ = 1
    elif layout == "NC":
        w_, k_ = a_.shape if is_var else get_const_tuple(a_.shape)
        n_, _ = b_.shape if is_var else get_const_tuple(b_.shape)
        h_ = 1
    else:
        raise ValueError("not support layout: " + layout)

    ki = tvm.reduce_axis((0, k_), name="ki")
    type_map = {
        "int8": "int32",
        "uint8": "uint32",
        "float32": "float32",
        "float16": "float16",
    }
    acum_dtype = type_map[a_.dtype]
    attrs = {"ci_": ci_, "vh_": vh_, "vw_": vw_, "vc_": vc_, "ACUM_DTYPE": acum_dtype}

    if layout == "NCHW":
        h_div = h_ // vh_
        w_div = w_ // vw_
        n_div = n_ // vc_
        avshape = (batch, h_div, w_div, vh_, vw_, k_)
        bvshape = (n_div, k_, vc_)
        ovshape = (batch, n_div, h_div, w_div, vh_, vw_, vc_)

        a_vec = tvm.compute(
            avshape,
            lambda n, oh, ow, vh, vw, ci: a_[n][ci][oh * vh_ + vh][ow * vw_ + vw],
            name="a_vec",
        )
        b_vec = tvm.compute(
            bvshape, lambda oc, ci, vc: b_[oc * vc_ + vc][ci], name="b_vec"
        )

        ma = tvm.compute(
            ovshape,
            lambda n, oc, oh, ow, vh, vw, vc: tvm.sum(
                a_vec[n, oh, ow, vh, vw, ki].astype(out_dtype)
                * b_vec[oc, ki, vc].astype(out_dtype),
                axis=[ki],
            ),
            name="matmul",
            attrs=attrs,
        )

        if is_transpose:
            oshape = (batch, h_div * vh_, w_div * vw_, n_div * vc_)

            output = tvm.compute(
                oshape,
                lambda n, h, w, c: ma[n][c // vc_][h // vh_][w // vw_][h % vh_][w % vw_][
                    c % vc_
                ],
                name="output_unpack",
                tag="asm_matmul_output",
            )
        else:
            oshape = (batch, n_div * vc_, h_div * vh_, w_div * vw_)
            output = tvm.compute(
                oshape,
                lambda n, c, h, w: ma[n][c // vc_][h // vh_][w // vw_][h % vh_][w % vw_][
                    c % vc_
                ],
                name="output_unpack",
                tag="asm_matmul_output",
            )
    elif layout == "NCH":
        w_div = w_ // vw_
        n_div = n_ // vc_
        avshape = (batch, w_div, vw_, k_)
        bvshape = (n_div, k_, vc_)
        ovshape = (batch, n_div, w_div, vw_, vc_)
        oshape = (batch, n_div * vc_, w_div * vw_)

        a_vec = tvm.compute(
            avshape, lambda b, om, vw, ci: a_[b][ci][om * vw_ + vw], name="a_vec"
        )
        b_vec = tvm.compute(
            bvshape, lambda on, ci, vc: b_[on * vc_ + vc][ci], name="b_vec"
        )

        ma = tvm.compute(
            ovshape,
            lambda b, on, om, vm, vn: tvm.sum(
                a_vec[b, om, vm, ki].astype(out_dtype)
                * b_vec[on, ki, vn].astype(out_dtype),
                axis=[ki],
            ),
            name="matmul",
            attrs=attrs,
        )

        output = tvm.compute(
            oshape,
            lambda b, n, m: ma[b][n // vc_][m // vw_][m % vw_][n % vc_],
            name="output_unpack",
            tag="asm_matmul_output",
        )
    elif layout == "NC":
        w_div = w_ // vw_
        n_div = n_ // vc_
        avshape = (w_div, vw_, k_)
        bvshape = (n_div, k_, vc_)
        ovshape = (w_div, n_div, vw_, vc_)
        oshape = (w_div * vw_, n_div * vc_)

        a_vec = tvm.compute(
            avshape, lambda om, vw, ci: a_[om * vw_ + vw][ci], name="a_vec"
        )
        b_vec = tvm.compute(
            bvshape, lambda on, ci, vc: b_[on * vc_ + vc][ci], name="b_vec"
        )

        ma = tvm.compute(
            ovshape,
            lambda om, on, vm, vn: tvm.sum(
                a_vec[om, vm, ki].astype(out_dtype)
                * b_vec[on, ki, vn].astype(out_dtype),
                axis=[ki],
            ),
            name="matmul",
            attrs=attrs,
        )

        output = tvm.compute(
            oshape,
            lambda m, n: ma[m // vw_][n // vc_][m % vw_][n % vc_],
            name="output_unpack",
            tag="asm_matmul_output",
        )
    else:
        raise ValueError("not support layout: " + layout)

    return output


def intrin_conv(args):
    """intrin_conv is a conv inner interface"""
    (
        ndim,
        ci_,
        vh_,
        vw_,
        vc_,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        dtype,
        acum_dtype,
        opname,
        core_id,
    ) = args
    ci_ = tvm.var("ci_") if ci_ is None else ci_
    kvshape = (ci_, vc_)
    if ndim == 2:
        dvshape = (vw_, ci_)
        ovshape = (vw_, vc_)

        data_vec = tvm.placeholder(dvshape, name="a", dtype=dtype)
        kernel_vec = tvm.placeholder(kvshape, name="b", dtype=dtype)
        ci = tvm.reduce_axis((0, ci_), name="ci")
        conv = tvm.compute(
            ovshape,
            lambda vw, vc: tvm.sum(
                data_vec[vw, ci].astype(acum_dtype)
                * kernel_vec[ci, vc].astype(acum_dtype),
                axis=[ci],
            ),
            name="conv",
        )
    else:
        dvshape = (vh_, vw_, ci_)
        ovshape = (vh_, vw_, vc_)

        data_vec = tvm.placeholder(dvshape, name="a", dtype=dtype)
        kernel_vec = tvm.placeholder(kvshape, name="b", dtype=dtype)
        ci = tvm.reduce_axis((0, ci_), name="ci")
        conv = tvm.compute(
            ovshape,
            lambda vh, vw, vc: tvm.sum(
                data_vec[vh, vw, ci].astype(acum_dtype)
                * kernel_vec[ci, vc].astype(acum_dtype),
                axis=[ci],
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

    ab_ = tvm.decl_buffer(
        data_vec.shape, data_vec.dtype, name="a_", offset_factor=1, strides=stride_a
    )
    bb_ = tvm.decl_buffer(
        kernel_vec.shape, kernel_vec.dtype, name="b_", offset_factor=1, strides=stride_b
    )
    cb_ = tvm.decl_buffer(
        conv.shape, conv.dtype, name="C", offset_factor=1, strides=stride_c
    )

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            b_ = tvm.ir_builder.create()
            b_.emit(
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
                    core_id,
                )
            )
            return b_.get()

        return _body()

    return tvm.decl_tensor_intrin(
        conv.op, intrin_func, binds={data_vec: ab_, kernel_vec: bb_, conv: cb_}
    )


def _schedule_asm(cfg, s, a_vec, b_vec, mat, output, last):
    """schedule implementation"""
    is_transpose = 0 if not isinstance(cfg, dict) else cfg["is_transpose"]
    attrs = mat.op.attrs
    vh_, vw_, vc_ = (attrs["vh_"].value, attrs["vw_"].value, attrs["vc_"].value)

    # axis split and reorder
    if len(a_vec.shape) == 3:
        ow, oc = s[last].op.axis
        oc, vc = s[last].split(oc, vc_)
        ow, vw = s[last].split(ow, vw_)
        s[last].reorder(ow, oc, vw, vc)
        s[last].vectorize(vc)
        oh = ow = oc
    elif len(a_vec.shape) == 4:
        n, oc, ow, vw, vc = s[last].op.axis
        oc, vc = s[last].split(oc, vc_)
        ow, vw = s[last].split(ow, vw_)
        s[last].reorder(n, oc, ow, vw, vc)
    elif len(a_vec.shape) == 6:
        if is_transpose:
            n, oh, ow, oc = s[last].op.axis
        else:
            n, oc, oh, ow = s[last].op.axis
        oc, vc = s[last].split(oc, vc_)
        oh, vh = s[last].split(oh, vh_)
        ow, vw = s[last].split(ow, vw_)
        s[last].reorder(n, oc, oh, ow, vh, vw, vc)
    else:
        raise ValueError("not support a_vec: " + str(len(a_vec.shape)))
    if last != output and isinstance(output.op, tvm.tensor.ComputeOp):
        s[output].compute_inline()

    s[mat].compute_at(s[last], ow)
    s[mat].vectorize(s[mat].op.axis[-1])

    # mark parallel
    s[last].parallel(oh)

    if len(a_vec.shape) == 3:
        om, _, _ = s[a_vec].op.axis
        s[a_vec].compute_at(s[last], ow)
        s[a_vec].parallel(om)
    elif len(a_vec.shape) == 4:
        _, om, _, _ = s[a_vec].op.axis
        s[a_vec].compute_at(s[last], ow)
        s[a_vec].parallel(om)
    else:
        _, oh, _, _, _, _ = s[a_vec].op.axis
        s[a_vec].parallel(oh)
    s[a_vec].vectorize(s[a_vec].op.axis[-1])
    s[a_vec].compute_inline()

    oc, _, _ = s[b_vec].op.axis
    s[b_vec].parallel(oc)
    s[b_vec].vectorize(s[b_vec].op.axis[-1])
    s[b_vec].compute_inline()
    return s


def _matmul_schedule_asm(cfg, outs):
    """schedule_conv2d_nchw schedule implementation"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "asm_matmul_output" in op.tag:
            # schedule conv2d
            output = op.output(0)
            mat = op.input_tensors[0]

            sidx = 0
            if mat.op.input_tensors[0].name == "attr":
                sidx = 1
            a_vec = mat.op.input_tensors[sidx]
            b_vec = mat.op.input_tensors[sidx + 1]

            def recurs_inline(a_):
                if a_.op.input_tensors:
                    a1 = a_.op.input_tensors[0]
                    if a1.shape == a_.shape:
                        s[a1].compute_inline()
                    recurs_inline(a1)

            def recurs_inline_(a_):
                if isinstance(a_, tvm.tensor.ComputeOp):
                    if a_.op.input_tensors:
                        a1 = a_.op.input_tensors[0]
                        s[a1].compute_inline()
                        recurs_inline_(a1)

            recurs_inline_(a_vec)
            recurs_inline_(b_vec)

            _schedule_asm(cfg, s, a_vec, b_vec, mat, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s
