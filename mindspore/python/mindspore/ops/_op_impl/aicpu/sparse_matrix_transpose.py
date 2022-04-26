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

"""SparseMatrixTranspose op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sparse_matrix_transpose_op_info = AiCPURegOp("SparseMatrixTranspose") \
    .fusion_type("OPAQUE") \
    .attr("conjugate", "bool") \
    .input(0, "x_dense_shape", "required") \
    .input(1, "x_batch_pointers", "required") \
    .input(2, "x_row_pointers", "required") \
    .input(3, "x_col_indices", "required") \
    .input(4, "x_values", "required") \
    .output(0, "y_dense_shape", "required") \
    .output(1, "y_batch_pointers", "required") \
    .output(2, "y_row_pointers", "required") \
    .output(3, "y_col_indices", "required") \
    .output(4, "y_values", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I8_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.U8_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.U8_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I16_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.U16_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.U16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.U32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.U32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I64_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.U64_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.U64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F16_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F16_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F64_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.F64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.C64_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.C64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.C128_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.I32_Default, DataType.C128_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I8_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I8_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.U8_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.U8_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I16_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.U16_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.U16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I32_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.U32_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.U32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.U64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.U64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F16_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F16_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F32_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.F64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.C64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.C64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.C128_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.I64_Default, DataType.C128_Default) \
    .get_op_info()


@op_info_register(sparse_matrix_transpose_op_info)
def _sparse_matrix_transpose_aicpu():
    """SparseMatrixTranspose AiCPU register"""
    return
