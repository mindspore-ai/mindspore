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

"""FakeLearnedScaleQuantPerChannelGradDReduce op"""
import te.lang.cce
from te import tvm
from te.platform.fusion_manager import fusion_manager
from tbe.tvm.topi import generic
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType


fake_learned_scale_quant_perchannel_grad_d_reduce_op_info = TBERegOp("FakeLearnedScaleQuantPerChannelGradDReduce") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("fake_learned_scale_quant_perchannel_grad_d_reduce.so") \
    .compute_cost(10) \
    .kernel_name("fake_learned_scale_quant_perchannel_grad_d_reduce") \
    .partial_flag(True) \
    .attr("channel_axis", "optional", "int", "all") \
    .input(0, "dout_alpha", None, "required", None) \
    .output(0, "dalpha", True, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD) \
    .get_op_info()


@op_info_register(fake_learned_scale_quant_perchannel_grad_d_reduce_op_info)
def _fake_learned_scale_quant_perchannel_grad_d_reduce_tbe():
    """FakeLearnedScaleQuantPerChannelGradDReduce TBE register"""
    return


@fusion_manager.register("fake_learned_scale_quant_perchannel_grad_d_reduce")
def fake_learned_scale_quant_perchannel_grad_d_reduce_compute(dout_alpha_data, dout_alpha, channel_axis,
                                                              kernel_name="fake_learned_scale_quant_perchannel_"
                                                                          "grad_d_reduce"):
    """FakeLearnedScaleQuantPerChannelGradDReduce"""
    dout_alpha_shape = dout_alpha.get("shape")
    axis = list(range(len(dout_alpha_shape)))
    axis.remove(channel_axis)
    dalpha = te.lang.cce.sum(dout_alpha_data, axis, False)
    return dalpha


@util.check_input_type(dict, dict, int, str)
def fake_learned_scale_quant_perchannel_grad_d_reduce(dout_alpha, dalpha, channel_axis,
                                                      kernel_name="fake_learned_scale_quant_perchannel_grad_d_reduce"):
    """FakeLearnedScaleQuantPerChannelGradDReduce"""

    dout_alpha_shape = dout_alpha.get("shape")
    dout_alpha_dtype = dout_alpha.get("dtype")

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(dout_alpha_shape)
    util.check_tensor_shape_size(dout_alpha_shape)

    check_list = ["float32", 'float16']
    dout_alpha_dtype = dout_alpha_dtype.lower()
    util.check_dtype_rule(dout_alpha_dtype, check_list)

    dout_alpha_data = tvm.placeholder(dout_alpha_shape, name="dout_alpha", dtype=dout_alpha_dtype)
    res = fake_learned_scale_quant_perchannel_grad_d_reduce_compute(dout_alpha_data, dout_alpha,
                                                                    channel_axis, kernel_name)

    with tvm.target.cce():
        sch = generic.auto_schedule(res)

    tensor_list = [dout_alpha_data, res]
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    te.lang.cce.cce_build_code(sch, config)
