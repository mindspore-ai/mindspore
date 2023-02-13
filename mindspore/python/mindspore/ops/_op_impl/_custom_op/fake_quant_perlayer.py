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

"""FakeQuantPerLayer op"""
from __future__ import absolute_import

from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

fake_quant_per_layer_op_info = TBERegOp("FakeQuantPerLayer") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("fake_quant_per_layer.so") \
    .compute_cost(10) \
    .kernel_name("fake_quant_per_layer") \
    .partial_flag(True) \
    .attr("symmetric", "optional", "bool", "all") \
    .attr("narrow_range", "optional", "bool", "all") \
    .attr("num_bits", "optional", "int", "all") \
    .input(0, "x", None, "required", None) \
    .input(1, "min", None, "required", None) \
    .input(2, "max", None, "required", None) \
    .output(0, "y", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(fake_quant_per_layer_op_info)
def _fake_quant_per_layer_tbe():
    """FakeQuantPerLayer TBE register"""
    return


@fusion_manager.register("fake_quant_per_layer")
def fake_quant_per_layer_compute(x, min_val, max_val, y, quant_min, quant_max, symmetric,
                                 kernel_name="fake_quant_per_layer"):
    """FakeQuantPerLayer"""
    shape = te.lang.cce.util.shape_to_list(x.shape)
    shape_min = te.lang.cce.util.shape_to_list(min_val.shape)
    quant_min = te.lang.cce.broadcast(quant_min, shape_min, x.dtype)
    quant_max = te.lang.cce.broadcast(quant_max, shape_min, x.dtype)
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

    # broadcast to shape
    nudge_min = te.lang.cce.broadcast(nudge_min, shape, x.dtype)
    nudge_max = te.lang.cce.broadcast(nudge_max, shape, x.dtype)
    scale = te.lang.cce.broadcast(scale, shape, x.dtype)

    # FakeQuant
    input_x = te.lang.cce.vmin(nudge_max, te.lang.cce.vmax(nudge_min, x))
    nudge_input_ = te.lang.cce.vdiv(
        te.lang.cce.vsub(input_x, nudge_min), scale)
    nudge_input = te.lang.cce.floor(te.lang.cce.vadds(nudge_input_, 0.5))
    res = te.lang.cce.vadd(te.lang.cce.vmul(nudge_input, scale), nudge_min)

    return res


@util.check_input_type(dict, dict, dict, dict, bool, bool, int, str)
def fake_quant_per_layer(x, min_val, max_val, y,
                         symmetric, narrow_range, num_bits,
                         kernel_name="fake_quant_per_layer"):
    """FakeQuantPerLayer"""
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

    check_list = ["float32", "float16"]
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

    input_data = tvm.placeholder(input_shape, name="x", dtype=x_dtype)
    min_data = tvm.placeholder(shape_min, name="min_data", dtype=min_dtype)
    max_data = tvm.placeholder(shape_min, name="max_data", dtype=max_dtype)
    res = fake_quant_per_layer_compute(input_data, min_data, max_data, y,
                                       quant_min, quant_max, symmetric, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [input_data, min_data, max_data, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
