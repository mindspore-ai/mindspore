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

"""FakeQuantPerChannel op"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

fake_quant_perchannel_op_info = TBERegOp("FakeQuantPerChannel") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("fake_quant_perchannel.so") \
    .compute_cost(10) \
    .kernel_name("fake_quant_perchannel") \
    .partial_flag(True) \
    .attr("symmetric", "optional", "bool", "all") \
    .attr("narrow_range", "optional", "bool", "all") \
    .attr("num_bits", "optional", "int", "all") \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "x", None, "required", None) \
    .input(1, "min", None, "required", None) \
    .input(2, "max", None, "required", None) \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .get_op_info()


@op_info_register(fake_quant_perchannel_op_info)
def _fake_quant_perchannel_tbe():
    """FakeQuantPerChannel TBE register"""
    return


@fusion_manager.register("fake_quant_perchannel")
def fake_quant_perchannel_compute(x, min_val, max_val, y, quant_min, quant_max, symmetric,
                                  kernel_name="fake_quant_perchannel"):
    """FakeQuantPerChannel"""
    x_shape = te.lang.cce.util.shape_to_list(x.shape)
    minmax_shape = te.lang.cce.util.shape_to_list(min_val.shape)
    quant_min = tvm.const(quant_min, x.dtype)
    quant_max = tvm.const(quant_max, x.dtype)
    quant_min = te.lang.cce.broadcast(quant_min, minmax_shape, x.dtype)
    quant_max = te.lang.cce.broadcast(quant_max, minmax_shape, x.dtype)
    if symmetric:
        max_val = te.lang.cce.vmax(te.lang.cce.vmuls(min_val, -1.), max_val)
        min_val = te.lang.cce.vmuls(max_val, -1.)

    scale = te.lang.cce.vdiv(te.lang.cce.vsub(
        max_val, min_val), te.lang.cce.vsub(quant_max, quant_min))
    zp_from_min = te.lang.cce.vsub(quant_min, te.lang.cce.vdiv(min_val, scale))

    # Nudge zero point
    nudge_zp_ = te.lang.cce.vmin(
        quant_max, te.lang.cce.vmax(quant_min, zp_from_min))
    nudge_zp = te.lang.cce.floor(te.lang.cce.vadds(nudge_zp_, 0.5))
    nudge_min = te.lang.cce.vmul(te.lang.cce.vsub(quant_min, nudge_zp), scale)
    nudge_max = te.lang.cce.vmul(te.lang.cce.vsub(quant_max, nudge_zp), scale)

    # FakeQuant
    nudge_min_b = te.lang.cce.broadcast(nudge_min, x_shape)
    nudge_max_b = te.lang.cce.broadcast(nudge_max, x_shape)
    scale_b = te.lang.cce.broadcast(scale, x_shape)

    input_x = te.lang.cce.vmin(nudge_max_b, te.lang.cce.vmax(nudge_min_b, x))
    nudge_input_ = te.lang.cce.vdiv(
        te.lang.cce.vsub(input_x, nudge_min_b), scale_b)
    nudge_input = te.lang.cce.floor(te.lang.cce.vadds(nudge_input_, 0.5))
    res = te.lang.cce.vadd(te.lang.cce.vmul(nudge_input, scale_b), nudge_min_b)

    return res


def fake_quant_perchannel_param(x, min_val, max_val, channel_axis,
                                kernel_name="fake_quant_perchannel"):
    """Get and check fake_quant_perchannel parameters"""
    x_shape = x.get("shape")
    x_shape_ = x.get("ori_shape")
    x_format = x.get("format")
    x_dtype = x.get("dtype")
    min_shape = min_val.get("ori_shape")
    min_dtype = min_val.get("dtype")
    max_shape = max_val.get("ori_shape")
    max_dtype = max_val.get("dtype")
    # for Dense weight quant, 2d[co,ci] -> 4d[1,co,ci,1], channel_axis_ need change to 1.
    if channel_axis == 0 and x_shape_[0] != min_shape[0] and x_shape_[1] == min_shape[0]:
        channel_axis_ = 1
    else:
        channel_axis_ = channel_axis
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(x_shape)
    util.check_shape_rule(min_shape, 1, 1, x_shape_[channel_axis_])
    util.check_shape_rule(max_shape, 1, 1, x_shape_[channel_axis_])
    util.check_tensor_shape_size(x_shape)
    util.check_tensor_shape_size(min_shape)
    util.check_tensor_shape_size(max_shape)

    check_list = ["float32", "float16"]
    x_dtype = x_dtype.lower()
    min_dtype = min_dtype.lower()
    max_dtype = max_dtype.lower()
    util.check_dtype_rule(x_dtype, check_list)
    util.check_dtype_rule(min_dtype, check_list)
    util.check_dtype_rule(max_dtype, check_list)

    shape_c = [1] * len(x_shape)
    shape_c[channel_axis_] = min_val.get("ori_shape")[0]
    if x_format == "NC1HWC0" and channel_axis_ == 1:
        shape_c = min_val.get("shape")
    return x_shape, shape_c, x_dtype


@util.check_input_type(dict, dict, dict, dict, bool, bool, int, int, str)
def fake_quant_perchannel(x, min_val, max_val, y,
                          symmetric, narrow_range, num_bits, channel_axis,
                          kernel_name="fake_quant_perchannel"):
    """FakeQuantPerChannel"""
    quant_min = 0
    quant_max = 2 ** num_bits - 1
    if narrow_range:
        quant_min = quant_min + 1

    x_shape, shape_c, x_dtype = fake_quant_perchannel_param(x, min_val, max_val,
                                                            channel_axis, kernel_name)
    input_data = tvm.placeholder(x_shape, name="x", dtype=x_dtype)
    min_data = tvm.placeholder(shape_c, name="min_val", dtype=x_dtype)
    max_data = tvm.placeholder(shape_c, name="max_val", dtype=x_dtype)
    res = fake_quant_perchannel_compute(input_data, min_data, max_data, y,
                                        quant_min, quant_max, symmetric, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [input_data, min_data, max_data, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
