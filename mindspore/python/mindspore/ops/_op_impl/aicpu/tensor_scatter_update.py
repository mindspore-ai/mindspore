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

"""TensorScatterUpdate op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

tensor_scatter_update_op_info = AiCPURegOp("TensorScatterUpdate") \
    .fusion_type("OPAQUE") \
    .input(0, "input_x", "required") \
    .input(1, "indices", "required") \
    .input(2, "updates", "required") \
    .output(0, "y", "required") \
    .dtype_format(DataType.I8_Default, DataType.I32_Default, DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I16_Default, DataType.I32_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.I32_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.U16_Default, DataType.I32_Default, DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.U32_Default, DataType.I32_Default, DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.U64_Default, DataType.I32_Default, DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I32_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I32_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.C64_Default, DataType.I32_Default, DataType.C64_Default, DataType.C64_Default) \
    .dtype_format(DataType.C128_Default, DataType.I32_Default, DataType.C128_Default, DataType.C128_Default) \
    .dtype_format(DataType.I8_Default, DataType.I64_Default, DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I16_Default, DataType.I64_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I64_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.U8_Default, DataType.I64_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.U16_Default, DataType.I64_Default, DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.U32_Default, DataType.I64_Default, DataType.U32_Default, DataType.U32_Default) \
    .dtype_format(DataType.U64_Default, DataType.I64_Default, DataType.U64_Default, DataType.U64_Default) \
    .dtype_format(DataType.F16_Default, DataType.I64_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.I64_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F64_Default, DataType.I64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.C64_Default, DataType.I64_Default, DataType.C64_Default, DataType.C64_Default) \
    .dtype_format(DataType.C128_Default, DataType.I64_Default, DataType.C128_Default, DataType.C128_Default) \
    .get_op_info()


@op_info_register(tensor_scatter_update_op_info)
def _tensor_scatter_update_aicpu():
    """TensorScatterUpdate AiCPU register"""
    return
