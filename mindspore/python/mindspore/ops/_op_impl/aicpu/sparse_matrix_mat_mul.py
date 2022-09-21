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

"""SparseMatrixMatMul op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

sparse_matrix_mat_mul_op_info = AiCPURegOp("SparseMatrixMatMul") \
    .fusion_type("OPAQUE") \
    .input(0, "x1_dense_shape", "required") \
    .input(1, "x1_batch_pointers", "required") \
    .input(2, "x1_row_pointers", "required") \
    .input(3, "x1_col_indices", "required") \
    .input(4, "x1_values", "required") \
    .input(5, "x2_dense", "required") \
    .output(0, "y_dense", "required") \
    .attr("transpose_x1", "bool") \
    .attr("transpose_x2", "bool") \
    .attr("adjoint_x1", "bool") \
    .attr("adjoint_x2", "bool") \
    .attr("transpose_output", "bool") \
    .attr("conjugate_output", "bool") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.C64_Default, DataType.C64_Default, DataType.C64_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.C128_Default, DataType.C128_Default, DataType.C128_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F64_Default, DataType.F64_Default, DataType.F64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.C64_Default, DataType.C64_Default, DataType.C64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.C128_Default, DataType.C128_Default, DataType.C128_Default) \
    .get_op_info()


@op_info_register(sparse_matrix_mat_mul_op_info)
def _sparse_matrix_mat_mul_aicpu():
    """SparseMatrixMatMul AiCPU register"""
    return
