
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

"""FakeQuantMinMaxPerChannelUpdate op"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from topi import generic
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType


fake_quant_min_max_per_channel_update_op_info = TBERegOp("FakeQuantMinMaxPerChannelUpdate") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fake_quant_min_max_per_channel_update.so") \
    .compute_cost(10) \
    .kernel_name("fake_quant_min_max_per_channel_update") \
    .partial_flag(True) \
    .attr("ema", "optional", "bool", "all") \
    .attr("ema_decay", "optional", "float", "all") \
    .attr("symmetric", "optional", "bool", "all") \
    .attr("narrow_range", "optional", "bool", "all") \
    .attr("training", "optional", "bool", "all") \
    .attr("num_bits", "optional", "int", "all") \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "x", None, "required", None) \
    .input(1, "min", None, "required", None) \
    .input(2, "max", None, "required", None) \
    .output(0, "min_up", True, "required", "all") \
    .output(1, "max_up", True, "required", "all") \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD,
                  DataType.F32_5HD) \
    .get_op_info()


@op_info_register(fake_quant_min_max_per_channel_update_op_info)
def _fake_quant_min_max_per_channel_update_tbe():
    """FakeQuantPerChannelUpdate TBE register"""
    return


@fusion_manager.register("fake_quant_min_max_per_channel_update")
def fake_quant_min_max_per_channel_update_compute(x, min_val, max_val,
                                                  ema, ema_decay, quant_min, quant_max, training, channel_axis,
                                                  kernel_name="fake_quant_min_max_per_channel_update"):
    """FakeQuantPerChannelUpdate compute"""
    shape_min = te.lang.cce.util.shape_to_list(min_val.shape)

    if not ema:
        ema_decay = 0.0
    if training:
        # CalMinMax
        axis = [0, 2, 3]
        x_min = te.lang.cce.reduce_min(x, axis=axis)
        x_max = te.lang.cce.reduce_max(x, axis=axis)
        x_min = te.lang.cce.broadcast(x_min, shape_min)
        x_max = te.lang.cce.broadcast(x_max, shape_min)
        min_val = te.lang.cce.vadd(te.lang.cce.vmuls(
            min_val, ema_decay), te.lang.cce.vmuls(x_min, (1 - ema_decay)))
        max_val = te.lang.cce.vadd(te.lang.cce.vmuls(
            max_val, ema_decay), te.lang.cce.vmuls(x_max, (1 - ema_decay)))
        min_val = te.lang.cce.vmins(min_val, 0)
        max_val = te.lang.cce.vmaxs(max_val, 0)

    return [min_val, max_val]


@util.check_input_type(dict, dict, dict, dict, dict, bool, float, bool, bool, bool, int, int, str)
def fake_quant_min_max_per_channel_update(x, min_val, max_val, min_up, max_up,
                                          ema, ema_decay, symmetric, narrow_range, training, num_bits, channel_axis,
                                          kernel_name="fake_quant_min_max_per_channel_update"):
    """FakeQuantPerLayer op"""
    x_shape = x.get("ori_shape")
    x_format = x.get("format")
    x_dtype = x.get("dtype")
    min_shape = min_val.get("ori_shape")
    min_dtype = min_val.get("dtype")
    max_shape = max_val.get("ori_shape")
    max_dtype = max_val.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(x_shape)
    util.check_shape_rule(min_shape, 1, 1, x_shape[channel_axis])
    util.check_shape_rule(max_shape, 1, 1, x_shape[channel_axis])
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

    if symmetric:
        quant_min = 0 - 2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1
    else:
        quant_min = 0
        quant_max = 2 ** num_bits - 1
    if narrow_range:
        quant_min = quant_min + 1

    shape_c = [min_val.get("shape")[1], min_val.get("shape")[-1]]
    input_data = tvm.placeholder(x.get("shape"), name="x", dtype=x_dtype)
    min_data = tvm.placeholder(shape_c, name="min_val", dtype=x_dtype)
    max_data = tvm.placeholder(shape_c, name="max_val", dtype=x_dtype)
    res_list = fake_quant_min_max_per_channel_update_compute(input_data, min_data, max_data,
                                                             ema, ema_decay, quant_min, quant_max, training, channel_axis, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res_list)

    tensor_list = [input_data, min_data, max_data] + list(res_list)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
