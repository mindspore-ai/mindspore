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

"""DenseToDenseSetOperation op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
dense_to_dense_set_operation_op_info = AiCPURegOp("DenseToDenseSetOperation") \
    .fusion_type("OPAQUE") \
    .attr("set_operation", "str")\
    .attr("validate_indices", "bool")\
    .input(0, "x1", "required") \
    .input(1, "x2", "required") \
    .output(0, "y_indices", "required") \
    .output(1, "y_values", "required") \
    .output(1, "y_shape", "required") \
    .dtype_format(DataType.I8_Default, DataType.I8_Default, DataType.I64_Default,
                  DataType.I8_Default, DataType.I64_Default) \
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.I64_Default,
                  DataType.I16_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I64_Default,
                  DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.U8_Default, DataType.I64_Default,
                  DataType.U8_Default, DataType.I64_Default) \
    .dtype_format(DataType.U16_Default, DataType.U16_Default, DataType.I64_Default,
                  DataType.U16_Default, DataType.I64_Default) \
    .get_op_info()


@op_info_register(dense_to_dense_set_operation_op_info)
def _dense_to_dense_set_operation_aicpu():
    """DenseToDenseSetOperation aicpu register"""
    return
