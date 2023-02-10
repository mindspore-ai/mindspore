# Copyright 2023 Huawei Technologies Co., Ltd
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

"""GenerateEodMask op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

generate_eod_mask_op_info = AiCPURegOp("GenerateEodMask") \
    .fusion_type("OPAQUE") \
    .attr("eod_token_id", "int") \
    .input(0, "inputs_ids", "required") \
    .output(0, "position_ids", "required") \
    .output(1, "attention_mask", "required") \
    .dtype_format(DataType.U16_Default, DataType.U16_Default, DataType.F16_Default) \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.F16_Default) \
    .dtype_format(DataType.U64_Default, DataType.U64_Default, DataType.F16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.F16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(generate_eod_mask_op_info)
def _generate_eod_mask_aicpu():
    """GenerateEodMask AiCPU register"""
    return
