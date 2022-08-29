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

"""SetSize op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

set_size_op_info = AiCPURegOp("SetSize")                                                                  \
    .fusion_type("OPAQUE")                                                                                \
    .input(0, "set_indices", "required")                                                                  \
    .input(1, "set_values", "required")                                                                   \
    .input(2, "set_shape", "required")                                                                    \
    .output(0, "size", "required")                                                                        \
    .attr("validate_indices", "bool")                                                                     \
    .dtype_format(DataType.I64_Default, DataType.I8_Default, DataType.I64_Default, DataType.I32_Default)  \
    .dtype_format(DataType.I64_Default, DataType.I16_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.U8_Default, DataType.I64_Default, DataType.I32_Default)  \
    .dtype_format(DataType.I64_Default, DataType.U16_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(set_size_op_info)
def _set_size_aicpu():
    """SetSize AiCPU register"""
    return
