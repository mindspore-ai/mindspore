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

"""FakeQuantPerLayerGrad op"""
from __future__ import absolute_import

from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

SHAPE_SIZE_LIMIT = 2147483648
D_TYPE = 'float32'

fake_quant_per_layer_grad_op_info = TBERegOp("FakeQuantPerLayerGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fake_quant_per_layer_grad.so") \
    .compute_cost(10) \
    .kernel_name("fake_quant_per_layer_grad") \
    .partial_flag(True) \
    .attr("num_bits", "optional", "int", "all") \
    .attr("symmetric", "optional", "bool", "all") \
    .attr("narrow_range", "optional", "bool", "all") \
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
    shape_inputs = te.lang.cce.util.shape_to_list(data_x.shape)
    min_value = tvm.const(2 ** (-126), dtype=D_TYPE)
    max_value = tvm.const(2 ** 62, dtype=D_TYPE)
    factor_value = tvm.const(2 ** 2, dtype=D_TYPE)
    data_zero = te.lang.cce.broadcast(
        tvm.const(0, dtype=D_TYPE), shape_inputs, D_TYPE)
    min_value_tensor = te.lang.cce.vadds(data_zero, min_value)

    res_sub = te.lang.cce.vsub(data_y, data_x)
    res_min = te.lang.cce.vmin(res_sub, min_value_tensor)
    res_max = te.lang.cce.vmax(res_min, data_zero)

    res_max_mul = te.lang.cce.vmuls(res_max, max_value)
    res_max_mul_max = te.lang.cce.vmuls(res_max_mul, max_value)
    res = te.lang.cce.vmuls(res_max_mul_max, factor_value)

    return res


@op_info_register(fake_quant_per_layer_grad_op_info)
def _fake_quant_per_layer_grad_tbe():
    """FakeQuantPerLayerGrad TBE register"""
    return


@fusion_manager.register("fake_quant_per_layer_grad")
def fake_quant_per_layer_grad_compute(dout, x, min_val, max_val, quant_min, quant_max, symmetric,
                                      kernel_name="fake_quant_per_layer_grad"):
    """FakeQuantPerLayerGrad"""
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_min = te.lang.cce.util.shape_to_list(min_val.shape)
    quant_min = tvm.const(quant_min, x.dtype)
    quant_max = tvm.const(quant_max, x.dtype)
    quant_min = te.lang.cce.broadcast(quant_min, shape_min)
    quant_max = te.lang.cce.broadcast(quant_max, shape_min)

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
    nudge_min = te.lang.cce.broadcast(nudge_min, shape)
    nudge_max = te.lang.cce.broadcast(nudge_max, shape)

    bool_over_min = _less_compare_float32(nudge_min, x)
    bool_less_max = _less_compare_float32(x, nudge_max)
    bool_between = te.lang.cce.vmul(bool_over_min, bool_less_max)
    res = te.lang.cce.vmul(dout, bool_between)

    return res


@util.check_input_type(dict, dict, dict, dict, dict, int, bool, bool, str)
def fake_quant_per_layer_grad(dout, x, min_val, max_val, dx,
                              num_bits, symmetric, narrow_range,
                              kernel_name="fake_quant_per_layer_grad"):
    """FakeQuantPerLayerGrad"""
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    min_shape = min_val.get("ori_shape")
    min_dtype = min_val.get("dtype")
    max_shape = max_val.get("ori_shape")
    max_dtype = max_val.get("dtype")

    min_shape = util.scalar2tensor_one(min_shape)
    max_shape = util.scalar2tensor_one(max_shape)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_shape_rule(min_shape, 1, 1, 1)
    util.check_shape_rule(max_shape, 1, 1, 1)
    util.check_tensor_shape_size(input_shape)
    util.check_tensor_shape_size(min_shape)
    util.check_tensor_shape_size(max_shape)

    check_list = ["float32", 'float16']
    x_dtype = input_dtype.lower()
    min_dtype = min_dtype.lower()
    max_dtype = max_dtype.lower()
    util.check_dtype_rule(x_dtype, check_list)
    util.check_dtype_rule(min_dtype, check_list)
    util.check_dtype_rule(max_dtype, check_list)

    input_shape = (functools_reduce(lambda x, y: x * y, input_shape[:]),)
    shape_min, _, _ = util.produce_shapes(min_shape, input_shape)

    quant_min = 0
    quant_max = 2 ** num_bits - 1
    if narrow_range:
        quant_min = quant_min + 1

    dout_data = tvm.placeholder(input_shape, name="dout", dtype=x_dtype)
    input_data = tvm.placeholder(input_shape, name="x", dtype=x_dtype)
    min_data = tvm.placeholder(shape_min, name="min_data", dtype=min_dtype)
    max_data = tvm.placeholder(shape_min, name="max_data", dtype=max_dtype)
    res = fake_quant_per_layer_grad_compute(dout_data, input_data, min_data, max_data,
                                            quant_min, quant_max, symmetric, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [dout_data, input_data, min_data, max_data, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
