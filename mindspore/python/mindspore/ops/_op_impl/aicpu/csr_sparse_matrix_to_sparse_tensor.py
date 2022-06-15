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

"""CSRSparseMatrixToSparseTensor op"""
from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType

csr_sparse_matrix_to_sparse_tensor_op_info = AiCPURegOp("CSRSparseMatrixToSparseTensor") \
    .fusion_type("OPAQUE") \
    .input(0, "x_dense_shape", "required") \
    .input(1, "x_batch_pointers", "required") \
    .input(2, "x_row_pointers", "required") \
    .input(3, "x_col_indices", "required") \
    .input(4, "x_values", "required") \
    .output(0, "indices", "required") \
    .output(1, "values", "required") \
    .output(2, "dense_shape", "required") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F32_Default, DataType.I32_Default, DataType.F32_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.F64_Default, DataType.I32_Default, DataType.F64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.C64_Default, DataType.I32_Default, DataType.C64_Default, DataType.I32_Default) \
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default, DataType.I32_Default,
                  DataType.C128_Default, DataType.I32_Default, DataType.C128_Default, DataType.I32_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F32_Default, DataType.I64_Default, DataType.F32_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.F64_Default, DataType.I64_Default, DataType.F64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.C64_Default, DataType.I64_Default, DataType.C64_Default, DataType.I64_Default) \
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default, DataType.I64_Default,
                  DataType.C128_Default, DataType.I64_Default, DataType.C128_Default, DataType.I64_Default) \
    .get_op_info()


@op_info_register(csr_sparse_matrix_to_sparse_tensor_op_info)
def _csr_sparse_matrix_to_sparse_tensor_aicpu():
    """CSRSparseMatrixToSparseTensor AiCPU register"""
    return
