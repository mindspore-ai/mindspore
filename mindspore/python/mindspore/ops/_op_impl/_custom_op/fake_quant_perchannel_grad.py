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

"""FakeQuantPerChannelGrad op"""
from __future__ import absolute_import

import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

SHAPE_SIZE_LIMIT = 2147483648
D_TYPE = 'float32'

fake_quant_perchannel_grad_op_info = TBERegOp("FakeQuantPerChannelGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fake_quant_perchannel_grad.so") \
    .compute_cost(10) \
    .kernel_name("fake_quant_perchannel_grad") \
    .partial_flag(True) \
    .attr("symmetric", "optional", "bool", "all") \
    .attr("narrow_range", "optional", "bool", "all") \
    .attr("num_bits", "optional", "int", "all") \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "dout", None, "required", None) \
    .input(1, "x", None, "required", None) \
    .input(2, "min", None, "required", None) \
    .input(3, "max", None, "required", None) \
    .output(0, "dx", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


def _less_compare_float32(data_x, data_y):
    """_less_compare_float32 compute"""
    input_shape = te.lang.cce.util.shape_to_list(data_x.shape)
    min_value = tvm.const(2 ** (-126), dtype=D_TYPE)
    max_value = tvm.const(2 ** 62, dtype=D_TYPE)
    factor_value = tvm.const(2 ** 2, dtype=D_TYPE)
    data_zero = te.lang.cce.broadcast(
        tvm.const(0, dtype=D_TYPE), input_shape, D_TYPE)
    min_value_tensor = te.lang.cce.vadds(data_zero, min_value)

    res_sub = te.lang.cce.vsub(data_y, data_x)
    res_min = te.lang.cce.vmin(res_sub, min_value_tensor)
    res_max = te.lang.cce.vmax(res_min, data_zero)

    res_max_mul = te.lang.cce.vmuls(res_max, max_value)
    res_max_mul_max = te.lang.cce.vmuls(res_max_mul, max_value)
    res = te.lang.cce.vmuls(res_max_mul_max, factor_value)

    return res


@op_info_register(fake_quant_perchannel_grad_op_info)
def _fake_quant_perchannel_grad_tbe():
    """FakeQuantPerChannelGrad TBE register"""
    return


@fusion_manager.register("fake_quant_perchannel_grad")
def fake_quant_perchannel_grad_compute(dout, x, min_val, max_val, quant_min, quant_max,
                                       kernel_name="fake_quant_perchannel_grad"):
    """FakeQuantPerChannelGrad"""
    x_shape = te.lang.cce.util.shape_to_list(x.shape)
    minmax_shape = te.lang.cce.util.shape_to_list(min_val.shape)
    quant_min = tvm.const(quant_min, x.dtype)
    quant_max = tvm.const(quant_max, x.dtype)
    quant_min = te.lang.cce.broadcast(quant_min, minmax_shape, x.dtype)
    quant_max = te.lang.cce.broadcast(quant_max, minmax_shape, x.dtype)

    scale = te.lang.cce.vdiv(te.lang.cce.vsub(
        max_val, min_val), te.lang.cce.vsub(quant_max, quant_min))
    zp_from_min = te.lang.cce.vsub(quant_min, te.lang.cce.vdiv(min_val, scale))

    # Nudge zero point
    nudge_zp_ = te.lang.cce.vmin(
        quant_max, te.lang.cce.vmax(quant_min, zp_from_min))
    nudge_zp = te.lang.cce.floor(te.lang.cce.vadds(nudge_zp_, 0.5))
    nudge_min = te.lang.cce.vmul(te.lang.cce.vsub(quant_min, nudge_zp), scale)
    nudge_max = te.lang.cce.vmul(te.lang.cce.vsub(quant_max, nudge_zp), scale)

    # FakeQuant Grad
    nudge_min_b = te.lang.cce.broadcast(nudge_min, x_shape)
    nudge_max_b = te.lang.cce.broadcast(nudge_max, x_shape)

    bool_over_min = _less_compare_float32(nudge_min_b, x)
    bool_less_max = _less_compare_float32(x, nudge_max_b)
    bool_between = te.lang.cce.vmul(bool_over_min, bool_less_max)
    res = te.lang.cce.vmul(dout, bool_between)

    return res


def fake_quant_perchannel_grad_param(x, min_val, max_val, channel_axis,
                                     kernel_name="fake_quant_perchannel_grad"):
    """Get and check FakeQuantPerChannelGrad parameters"""
    x_shape = x.get("shape")
    x_shape_ = x.get("ori_shape")
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
    if shape_c[channel_axis_] != x_shape[channel_axis_]:
        shape_c = min_val.get("shape")
    return x_shape, shape_c, x_dtype


@util.check_input_type(dict, dict, dict, dict, dict, bool, bool, int, int, str)
def fake_quant_perchannel_grad(dout, x, min_val, max_val, dx,
                               symmetric, narrow_range, num_bits, channel_axis,
                               kernel_name="fake_quant_perchannel_grad"):
    """FakeQuantPerChannelGrad"""
    if symmetric:
        quant_min = 0 - 2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1
    else:
        quant_min = 0
        quant_max = 2 ** num_bits - 1
    if narrow_range:
        quant_min = quant_min + 1

    x_shape, shape_c, x_dtype = fake_quant_perchannel_grad_param(x, min_val, max_val,
                                                                 channel_axis, kernel_name)
    dout_data = tvm.placeholder(x_shape, name="dout", dtype=x_dtype)
    input_data = tvm.placeholder(x_shape, name="x", dtype=x_dtype)
    min_data = tvm.placeholder(shape_c, name="min_val", dtype=x_dtype)
    max_data = tvm.placeholder(shape_c, name="max_val", dtype=x_dtype)
    res = fake_quant_perchannel_grad_compute(dout_data, input_data, min_data, max_data,
                                             quant_min, quant_max, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [dout_data, input_data, min_data, max_data, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
