# Copyright 2022 Huawei Technologies Co., Ltd
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

"""CTCLossV2 op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
ctc_loss_v2_op_info = AiCPURegOp("CTCLossV2") \
    .fusion_type("OPAQUE") \
    .input(0, "log_probs", "required") \
    .input(1, "targets", "required") \
    .input(2, "input_lengths", "required") \
    .input(3, "target_lengths", "required") \
    .output(0, "neg_log_likelihood", "required") \
    .output(1, "log_alpha", "required") \
    .attr("blank", "int") \
    .attr("reduction", "str") \
    .attr("zero_infinity", "bool") \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(ctc_loss_v2_op_info)
def _ctc_loss_v2_aicpu():
    """CTCLossV2 AiCPU register"""
    return
