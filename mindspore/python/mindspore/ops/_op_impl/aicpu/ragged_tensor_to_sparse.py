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

"""RaggedTensorToSparse op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
ragged_tensor_to_sparse_op_info = AiCPURegOp("RaggedTensorToSparse") \
    .fusion_type("OPAQUE") \
    .input(0, "rt_nested_splits", "dynamic") \
    .input(1, "rt_dense_values", "required") \
    .output(0, "sparse_indices", "required") \
    .output(1, "sparse_values", "required") \
    .output(2, "sparse_dense_shape", "required") \
    .attr("RAGGED_RANK", "int") \
    .attr("Tsplits", "Type") \
    .dtype_format(DataType.I32_Default, DataType.BOOL_Default, DataType.I64_Default, DataType.BOOL_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I8_Default, DataType.I64_Default, DataType.I8_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.U8_Default, DataType.I64_Default, DataType.U8_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I16_Default, DataType.I64_Default, DataType.I16_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.U16_Default, DataType.I64_Default, DataType.U16_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I64_Default, DataType.I32_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.F16_Default, DataType.I64_Default, DataType.F16_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.F32_Default, DataType.I64_Default, DataType.F32_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.F64_Default, DataType.I64_Default, DataType.F64_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.BOOL_Default, DataType.I64_Default, DataType.BOOL_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I8_Default, DataType.I64_Default, DataType.I8_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.U8_Default, DataType.I64_Default, DataType.U8_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I16_Default, DataType.I64_Default, DataType.I16_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.U16_Default, DataType.I64_Default, DataType.U16_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I32_Default, DataType.I64_Default, DataType.I32_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.F16_Default, DataType.I64_Default, DataType.F16_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.F32_Default, DataType.I64_Default, DataType.F32_Default, \
    DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.F64_Default, DataType.I64_Default, DataType.F64_Default, \
    DataType.I64_Default) \
    .get_op_info()


@op_info_register(ragged_tensor_to_sparse_op_info)
def _ragged_tensor_to_sparse_aicpu():
    """RaggedTensorToSparse AiCPU register"""
    return
