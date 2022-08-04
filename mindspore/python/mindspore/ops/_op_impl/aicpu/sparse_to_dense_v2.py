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

"""SparseToDense op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sparse_to_dense_v2_op_info = AiCPURegOp("SparseToDenseV2") \
    .fusion_type("OPAQUE") \
    .attr("validate_indices", "bool") \
    .input(0, "indices", "required") \
    .input(1, "output_shape", "required") \
    .input(2, "values", "required") \
    .input(3, "default_value", "required") \
    .output(0, "y", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.I8_Default, DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.I16_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.U8_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.U16_Default, DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.I8_Default, DataType.I8_Default, DataType.I8_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.I16_Default, DataType.I16_Default, DataType.I16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.I32_Default, DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.I64_Default, DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.U8_Default, DataType.U8_Default, DataType.U8_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.U16_Default, DataType.U16_Default, DataType.U16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, \
    DataType.BOOL_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, \
    DataType.BOOL_Default, DataType.BOOL_Default, DataType.BOOL_Default) \
    .get_op_info()


@op_info_register(sparse_to_dense_v2_op_info)
def _sparse_to_dense_v2_aicpu():
    """SparseToDenseV2 AiCPU register"""
    return
