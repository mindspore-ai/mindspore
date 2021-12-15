# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed unde:q!r the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""SigmoidCrossEntropyWithLogits op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

sigmoid_cross_entropy_with_logits_op_info = TBERegOp("SigmoidCrossEntropyWithLogits") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("sigmoid_cross_entropy_with_logits.so") \
    .compute_cost(10) \
    .kernel_name("sigmoid_cross_entropy_with_logits") \
    .partial_flag(True) \
    .input(0, "predict", False, "required", "all") \
    .input(1, "target", False, "required", "all") \
    .output(0, "loss", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD, DataType.F16_5HD) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F32_5HD, DataType.F32_5HD, DataType.F32_5HD) \
    .dtype_format(DataType.F16_NDC1HWC0, DataType.F16_NDC1HWC0, DataType.F16_NDC1HWC0) \
    .dtype_format(DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0, DataType.F32_NDC1HWC0) \
    .get_op_info()


@op_info_register(sigmoid_cross_entropy_with_logits_op_info)
def _sigmoid_cross_entropy_with_logits_tbe():
    """SigmoidCrossEntropyWithLogits TBE register"""
    return
