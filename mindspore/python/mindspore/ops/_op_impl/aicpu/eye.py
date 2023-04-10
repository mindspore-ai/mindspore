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
"""Eye op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

eye_op_info = AiCPURegOp("Eye")                                                \
    .fusion_type("OPAQUE")                                                             \
    .output(0, "output", "required")                                                   \
    .attr("num_rows", "int") \
    .attr("num_columns", "int") \
    .attr("dtype", "int") \
    .dtype_format(DataType.BOOL_Default) \
    .dtype_format(DataType.I8_Default) \
    .dtype_format(DataType.I16_Default) \
    .dtype_format(DataType.I32_Default) \
    .dtype_format(DataType.I64_Default) \
    .dtype_format(DataType.U8_Default) \
    .dtype_format(DataType.U16_Default) \
    .dtype_format(DataType.U32_Default) \
    .dtype_format(DataType.U64_Default) \
    .dtype_format(DataType.F32_Default) \
    .dtype_format(DataType.F16_Default) \
    .dtype_format(DataType.F64_Default) \
    .dtype_format(DataType.C64_Default) \
    .dtype_format(DataType.C128_Default) \
    .get_op_info()


@op_info_register(eye_op_info)
def _eye_aicpu():
    """Eye aicpu register"""
    return
