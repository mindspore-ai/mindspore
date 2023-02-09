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

"""SequenceAdd op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sequence_add_op_info = AiCPURegOp("SequenceAdd") \
    .fusion_type("OPAQUE") \
    .input(0, "input_0", "required") \
    .input(1, "input_1", "required") \
    .output(0, "output_data", "required") \
    .dtype_format(DataType.F32_Default_Tuple, DataType.F32_Default_Tuple, DataType.F32_Default_Tuple) \
    .dtype_format(DataType.F64_Default_Tuple, DataType.F64_Default_Tuple, DataType.F64_Default_Tuple) \
    .dtype_format(DataType.I32_Default_Tuple, DataType.I32_Default_Tuple, DataType.I32_Default_Tuple) \
    .dtype_format(DataType.I64_Default_Tuple, DataType.I64_Default_Tuple, DataType.I64_Default_Tuple) \
    .get_op_info()


@op_info_register(sequence_add_op_info)
def _sequence_add_aicpu():
    """SequenceAdd AiCPU register"""
    return
