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

"""CTCLossV2Grad op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
ctc_loss_v2_grad_op_info = AiCPURegOp("CTCLossV2Grad") \
    .fusion_type("OPAQUE") \
    .input(0, "grad_out", "required") \
    .input(1, "log_probs", "required") \
    .input(2, "targets", "required") \
    .input(3, "input_lengths", "required") \
    .input(4, "target_lengths", "required") \
    .input(5, "neg_log_likelihood", "required") \
    .input(6, "log_alpha", "required") \
    .output(0, "grad", "required") \
    .attr("blank", "int") \
    .attr("reduction", "str") \
    .attr("zero_infinity", "bool") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.F64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .get_op_info()


@op_info_register(ctc_loss_v2_grad_op_info)
def _ctc_loss_v2_grad_aicpu():
    """CTCLossV2Grad AiCPU register"""
    return
