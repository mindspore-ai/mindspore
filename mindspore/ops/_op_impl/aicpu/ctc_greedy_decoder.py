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

"""CTCGreedyDecoder op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
ctc_greedy_decoder_op_info = AiCPURegOp("CTCGreedyDecoder") \
    .fusion_type("OPAQUE") \
    .input(0, "inputs", "required") \
    .input(1, "sequence_length", "required") \
    .output(0, "decoded_indices", "required") \
    .output(1, "decoded_values", "required") \
    .output(2, "decoded_shape", "required") \
    .output(3, "log_probability", "required") \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F64_Default) \
    .get_op_info()

@op_info_register(ctc_greedy_decoder_op_info)
def _ctc_greedy_decoder_aicpu():
    """CTCGreedyDecoder AiCPU register"""
    return
