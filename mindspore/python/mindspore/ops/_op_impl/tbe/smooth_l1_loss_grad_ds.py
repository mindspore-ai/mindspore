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

"""SmoothL1LossGrad op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

smooth_l1_loss_grad_op_info = TBERegOp("SmoothL1LossGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("smooth_l1_loss_grad_v2.so") \
    .compute_cost(10) \
    .kernel_name("smooth_l1_loss_grad_v2") \
    .partial_flag(True) \
    .dynamic_shape(True) \
    .attr("beta", "optional", "float", "all") \
    .attr("reduction", "optional", "str", "all") \
    .input(0, "predict", False, "required", "all") \
    .input(1, "label", False, "required", "all") \
    .input(2, "dout", False, "required", "all") \
    .output(0, "loss", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_FracZ, DataType.F16_FracZ) \
    .dtype_format(DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0, DataType.F16_C1HWNCoC0) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ, DataType.F32_FracZ) \
    .dtype_format(DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0, DataType.F32_C1HWNCoC0) \
    .get_op_info()


@op_info_register(smooth_l1_loss_grad_op_info)
def _smooth_l1_loss_grad_ds_tbe():
    """SmoothL1LossGrad TBE register"""
    return
