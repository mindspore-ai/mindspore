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

"""Greater op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

greater_op_info = AiCPURegOp("Greater")                                     \
    .fusion_type("OPAQUE")                                                  \
    .input(0, "x1", "required")                                             \
    .input(1, "x2", "required")                                             \
    .output(0, "y", "required")                                             \
    .dtype_format(DataType.I8_None, DataType.I8_None, DataType.BOOL_None)   \
    .dtype_format(DataType.I16_None, DataType.I16_None, DataType.BOOL_None) \
    .dtype_format(DataType.I32_None, DataType.I32_None, DataType.BOOL_None) \
    .dtype_format(DataType.I64_None, DataType.I64_None, DataType.BOOL_None) \
    .dtype_format(DataType.U8_None, DataType.U8_None, DataType.BOOL_None)   \
    .dtype_format(DataType.U16_None, DataType.U16_None, DataType.BOOL_None) \
    .dtype_format(DataType.U32_None, DataType.U32_None, DataType.BOOL_None) \
    .dtype_format(DataType.U64_None, DataType.U64_None, DataType.BOOL_None) \
    .dtype_format(DataType.F16_None, DataType.F16_None, DataType.BOOL_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.BOOL_None) \
    .dtype_format(DataType.F64_None, DataType.F64_None, DataType.BOOL_None) \
    .get_op_info()


@op_info_register(greater_op_info)
def _greater_aicpu():
    """Greater AICPU register"""
    return
