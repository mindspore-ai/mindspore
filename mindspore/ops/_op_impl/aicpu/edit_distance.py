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

"""EditDistance op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

edit_distance_op_info = AiCPURegOp("EditDistance") \
    .fusion_type("OPAQUE") \
    .input(0, "hypothesis_indices", "required") \
    .input(1, "hypothesis_values", "required") \
    .input(2, "hypothesis_shape", "required") \
    .input(3, "truth_indices", "required") \
    .input(4, "truth_values", "required") \
    .input(5, "truth_shape", "required") \
    .output(0, "y", "required") \
    .attr("normalize", "bool") \
    .dtype_format(DataType.I64_Default, DataType.I8_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.I8_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.I16_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.I16_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.I32_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.U8_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.U8_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.U16_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.U16_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.U32_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.U32_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.U64_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.U64_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.F16_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.F16_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.F32_Default, DataType.I64_Default, DataType.F32_Default,) \
    .dtype_format(DataType.I64_Default, DataType.F64_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.F64_Default, DataType.I64_Default, DataType.F32_Default,) \
    .get_op_info()

@op_info_register(edit_distance_op_info)
def _edit_distance_aicpu():
    """EditDistance AiCPU register"""
    return
