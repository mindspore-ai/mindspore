# Copyright 2021 Huawei Technologies Co., Ltd
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

"""FakeLearnedScaleQuantPerLayer op"""
from functools import reduce as functools_reduce
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

fake_learned_scale_quant_perlayer_op_info = TBERegOp("FakeLearnedScaleQuantPerLayer") \
    .fusion_type("ELEMWISE") \
    .async_flag(False) \
    .binfile_name("fake_learned_scale_quant_perlayer.so") \
    .compute_cost(10) \
    .kernel_name("fake_learned_scale_quant_perlayer") \
    .partial_flag(True) \
    .attr("neg_trunc", "optional", "bool", "all") \
    .input(0, "input_x", None, "required", None) \
    .input(1, "alpha", None, "required", None) \
    .input(2, "quant_max", None, "required", None) \
    .output(0, "out", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(fake_learned_scale_quant_perlayer_op_info)
def _fake_learned_scale_quant_perlayer_tbe():
    """FakeLearnedScaleQuantPerLayer TBE register"""
    return


@fusion_manager.register("fake_learned_scale_quant_perlayer")
def fake_learned_scale_quant_perlayer_compute(input_data, alpha_data, quant_max_data, neg_trunc,
                                              kernel_name="fake_learned_scale_quant_perlayer"):
    """FakeLearnedScaleQuantPerLayer"""
    input_shape = te.lang.cce.util.shape_to_list(input_data.shape)
    eps = tvm.const(1e-6, input_data.dtype)
    alpha_data = te.lang.cce.vcmpsel(te.lang.cce.vabs(alpha_data), eps, 'ge', alpha_data, eps)
    alpha_data = te.lang.cce.broadcast(alpha_data, input_shape, input_data.dtype)
    quant_max_data = te.lang.cce.broadcast(quant_max_data, input_shape, input_data.dtype)

    input_x = te.lang.cce.vdiv(input_data, alpha_data)

    if neg_trunc:
        input_x = te.lang.cce.round_to(input_x, 1.0, 0.0)
    else:
        input_x = te.lang.cce.round_to(input_x, 1.0, -1.0)

    nudge_input = te.lang.cce.floor(te.lang.cce.vadds(te.lang.cce.vmul(input_x, quant_max_data), 0.5))
    input_quant = te.lang.cce.vdiv(nudge_input, quant_max_data)
    res = te.lang.cce.vmul(input_quant, alpha_data)

    return res


def fake_learned_scale_quant_perlayer_param(input_x, alpha, quant_max,
                                            kernel_name="fake_learned_scale_quant_perlayer"):
    """Get and check FakeLearnedScaleQuantPerLayer parameters"""
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype")
    alpha_shape = alpha.get("ori_shape")
    alpha_dtype = alpha.get("dtype")
    quant_max_shape = quant_max.get("ori_shape")
    quant_max_dtype = quant_max.get("dtype")

    alpha_shape = util.scalar2tensor_one(alpha_shape)
    quant_max_shape = util.scalar2tensor_one(quant_max_shape)
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(input_shape)
    util.check_shape_rule(alpha_shape, 1, 1, 1)
    util.check_shape_rule(quant_max_shape, 1, 1, 1)
    util.check_tensor_shape_size(input_shape)
    util.check_tensor_shape_size(alpha_shape)
    util.check_tensor_shape_size(quant_max_shape)

    check_list = ["float32", "float16"]
    input_dtype = input_dtype.lower()
    alpha_dtype = alpha_dtype.lower()
    quant_max_dtype = quant_max_dtype.lower()
    util.check_dtype_rule(input_dtype, check_list)
    util.check_dtype_rule(alpha_dtype, check_list)
    util.check_dtype_rule(quant_max_dtype, check_list)

    input_shape = (functools_reduce(lambda x, y: x * y, input_shape[:]),)

    input_data = tvm.placeholder(input_shape, name="x", dtype=input_dtype)
    alpha_data = tvm.placeholder(alpha_shape, name="alpha_data", dtype=alpha_dtype)
    quant_max_data = tvm.placeholder(quant_max_shape, name="quant_max_data", dtype=quant_max_dtype)
    return input_data, alpha_data, quant_max_data


@util.check_input_type(dict, dict, dict, dict, bool, str)
def fake_learned_scale_quant_perlayer(input_x, alpha, quant_max, out, neg_trunc,
                                      kernel_name="fake_learned_scale_quant_perlayer"):
    """FakeLearnedScaleQuantPerLayer"""
    input_data, alpha_data, quant_max_data = \
        fake_learned_scale_quant_perlayer_param(input_x, alpha, quant_max, kernel_name)

    res = fake_learned_scale_quant_perlayer_compute(input_data, alpha_data, quant_max_data, neg_trunc, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [input_data, alpha_data, quant_max_data, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              "bool_storage_as_1bit": False}

    te.lang.cce.cce_build_code(sch, config)
