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

"""SequenceAddN op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sequence_addn_op_info = AiCPURegOp("SequenceAddN") \
    .fusion_type("OPAQUE") \
    .input(0, "input_0", "required") \
    .output(0, "output_data", "required") \
    .dtype_format(DataType.U32_Default_Tuple, DataType.U32_Default) \
    .dtype_format(DataType.U64_Default_Tuple, DataType.U64_Default) \
    .dtype_format(DataType.I64_Default_Tuple, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default_Tuple, DataType.I32_Default) \
    .dtype_format(DataType.F64_Default_Tuple, DataType.F64_Default) \
    .dtype_format(DataType.F32_Default_Tuple, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default_Tuple, DataType.F16_Default) \
    .dtype_format(DataType.C64_Default_Tuple, DataType.C64_Default) \
    .dtype_format(DataType.C128_Default_Tuple, DataType.C128_Default) \
    .get_op_info()


@op_info_register(sequence_addn_op_info)
def _sequence_addn_aicpu():
    """SequenceAddN AiCPU register"""
    return
