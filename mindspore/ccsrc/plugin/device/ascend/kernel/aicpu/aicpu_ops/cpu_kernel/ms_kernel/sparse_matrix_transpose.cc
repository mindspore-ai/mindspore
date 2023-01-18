/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * UnSparseMatrixTranspose required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sparse_matrix_transpose.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include <numeric>
#include <iostream>

using namespace std;

namespace aicpu {
const uint32_t kInputNum = 5;
const uint32_t kOutputNum = 5;
const uint32_t kzero = 0;
const uint32_t kone = 1;
const uint32_t ktwo = 2;
const uint32_t kthree = 3;
const uint32_t kfour = 4;
const uint32_t krankwithbatch = 3;
const char *SPARSEMATRIXTRANSPOSE = "SparseMatrixTranspose";
}  // namespace aicpu
namespace aicpu {
uint32_t SparseMatrixTransposeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SparseMatrixTranspose normal check failed.");
  DataType indice_type = ctx.Input(0)->GetDataType();
  DataType value_type = ctx.Input(4)->GetDataType();
  uint32_t status;
  switch (indice_type) {
    case DT_INT32:
      switch (value_type) {
        case DT_UINT8:
          status = SparseMatrixTransposeCompute<int32_t, uint8_t>(ctx);
          break;
        case DT_UINT16:
          status = SparseMatrixTransposeCompute<int32_t, uint16_t>(ctx);
          break;
        case DT_UINT32:
          status = SparseMatrixTransposeCompute<int32_t, uint32_t>(ctx);
          break;
        case DT_UINT64:
          status = SparseMatrixTransposeCompute<int32_t, uint64_t>(ctx);
          break;
        case DT_INT8:
          status = SparseMatrixTransposeCompute<int32_t, int8_t>(ctx);
          break;
        case DT_INT16:
          status = SparseMatrixTransposeCompute<int32_t, int16_t>(ctx);
          break;
        case DT_INT32:
          status = SparseMatrixTransposeCompute<int32_t, int32_t>(ctx);
          break;
        case DT_INT64:
          status = SparseMatrixTransposeCompute<int32_t, int64_t>(ctx);
          break;
        case DT_FLOAT16:
          status = SparseMatrixTransposeCompute<int32_t, Eigen::half>(ctx);
          break;
        case DT_FLOAT:
          status = SparseMatrixTransposeCompute<int32_t, float_t>(ctx);
          break;
        case DT_DOUBLE:
          status = SparseMatrixTransposeCompute<int32_t, double_t>(ctx);
          break;
        case DT_COMPLEX64:
          status = SparseMatrixTransposeComputecomplex<int32_t, complex<float_t>>(ctx);
          break;
        case DT_COMPLEX128:
          status = SparseMatrixTransposeComputecomplex<int32_t, complex<double_t>>(ctx);
          break;
        default:
          KERNEL_LOG_ERROR("data type of x_value is not required");
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (value_type) {
        case DT_UINT8:
          status = SparseMatrixTransposeCompute<int64_t, uint8_t>(ctx);
          break;
        case DT_UINT16:
          status = SparseMatrixTransposeCompute<int64_t, uint16_t>(ctx);
          break;
        case DT_UINT32:
          status = SparseMatrixTransposeCompute<int64_t, uint32_t>(ctx);
          break;
        case DT_UINT64:
          status = SparseMatrixTransposeCompute<int64_t, uint64_t>(ctx);
          break;
        case DT_INT8:
          status = SparseMatrixTransposeCompute<int64_t, int8_t>(ctx);
          break;
        case DT_INT16:
          status = SparseMatrixTransposeCompute<int64_t, int16_t>(ctx);
          break;
        case DT_INT32:
          status = SparseMatrixTransposeCompute<int64_t, int32_t>(ctx);
          break;
        case DT_INT64:
          status = SparseMatrixTransposeCompute<int64_t, int64_t>(ctx);
          break;
        case DT_FLOAT16:
          status = SparseMatrixTransposeCompute<int64_t, Eigen::half>(ctx);
          break;
        case DT_FLOAT:
          status = SparseMatrixTransposeCompute<int64_t, float_t>(ctx);
          break;
        case DT_DOUBLE:
          status = SparseMatrixTransposeCompute<int64_t, double_t>(ctx);
          break;
        case DT_COMPLEX64:
          status = SparseMatrixTransposeComputecomplex<int64_t, complex<float_t>>(ctx);
          break;
        case DT_COMPLEX128:
          status = SparseMatrixTransposeComputecomplex<int64_t, complex<double_t>>(ctx);
          break;
        default:
          KERNEL_LOG_ERROR("data type of x_value is not required");
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      KERNEL_LOG_ERROR("data type of dense shape is not int32 or int64");
      return KERNEL_STATUS_PARAM_INVALID;
  }

  if (status != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("error in do the actual compute!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename indiceT, typename valueT>
uint32_t SparseMatrixTransposeCpuKernel::SparseMatrixTransposeCompute(CpuKernelContext &ctx) {
  indiceT *x_dense_shape = static_cast<indiceT *>(ctx.Input(0)->GetData());
  indiceT *x_batch_pointers = static_cast<indiceT *>(ctx.Input(1)->GetData());
  indiceT *x_row_pointers = static_cast<indiceT *>(ctx.Input(2)->GetData());
  indiceT *x_col_indices = static_cast<indiceT *>(ctx.Input(3)->GetData());
  valueT *x_values = static_cast<valueT *>(ctx.Input(4)->GetData());
  bool conjugate = (ctx.GetAttr("conjugate") == nullptr) ? false : ctx.GetAttr("conjugate")->GetBool();
  indiceT *y_dense_shape = static_cast<indiceT *>(ctx.Output(0)->GetData());
  indiceT *y_batch_pointers = static_cast<indiceT *>(ctx.Output(1)->GetData());
  indiceT *y_row_pointers = static_cast<indiceT *>(ctx.Output(2)->GetData());
  indiceT *y_col_indices = static_cast<indiceT *>(ctx.Output(3)->GetData());
  valueT *y_values = static_cast<valueT *>(ctx.Output(4)->GetData());
  auto rank = ctx.Input(0)->NumElements();
  if (rank == krankwithbatch) {
    y_dense_shape[0] = x_dense_shape[0];
    y_dense_shape[1] = x_dense_shape[ktwo];
    y_dense_shape[ktwo] = x_dense_shape[1];
  } else {
    y_dense_shape[0] = x_dense_shape[1];
    y_dense_shape[1] = x_dense_shape[0];
  }
  auto batch_pointers = ctx.Input(1)->NumElements();
  for (int i = 0; i < batch_pointers; ++i) {
    y_batch_pointers[i] = x_batch_pointers[i];
  }

  auto num_rows = x_dense_shape[rank - 2];
  auto num_cols = x_dense_shape[rank - 1];
  auto num_batch = ctx.Input(1)->NumElements() - 1;
  int y_part_row_pointers[num_cols + 1];
  int part_row_pointers[num_rows + 1];

  for (int j = 0; j < num_batch; ++j) {
    int n = x_batch_pointers[j + 1] - x_batch_pointers[j];
    valueT part_values[n];
    indiceT part_col_indices[n];
    indiceT y_part_col_indices[n];
    valueT y_part_values[n];
    for (int i = 0; i < num_cols + 1; ++i) {
      y_part_row_pointers[i] = 0;
    }
    for (int k = 0; k < num_rows + 1; ++k) {
      part_row_pointers[k] = x_row_pointers[(num_rows + 1) * j + k];
    }
    for (int k = 0; k < n; ++k) {
      part_values[k] = x_values[x_batch_pointers[j] + k];
      part_col_indices[k] = x_col_indices[x_batch_pointers[j] + k];
    }
    for (int64_t i = 0; i < n; ++i) {
      y_part_row_pointers[part_col_indices[i] + 1] += 1;
    }
    std::partial_sum(y_part_row_pointers, y_part_row_pointers + num_cols + 1, y_part_row_pointers);
    for (int k = 0; k < num_cols + 1; ++k) {
      y_row_pointers[(num_cols + 1) * j + k] = y_part_row_pointers[k];
    }

    for (int k = 0; k < n; ++k) {
      part_values[k] = x_values[x_batch_pointers[j] + k];
      part_col_indices[k] = x_col_indices[x_batch_pointers[j] + k];
    }
    std::vector<int> current_col_count(num_cols);
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const int64_t row_begin = part_row_pointers[row_idx];
      const int64_t row_end = part_row_pointers[row_idx + 1];
      for (int64_t i = row_begin; i < row_end; ++i) {
        const int col_idx = part_col_indices[i];
        const int64_t offset = y_part_row_pointers[col_idx] + current_col_count[col_idx];
        y_part_col_indices[offset] = row_idx;
        y_part_values[offset] = part_values[i];
        current_col_count[col_idx] += 1;
      }
    }
    for (int k = 0; k < n; ++k) {
      y_values[x_batch_pointers[j] + k] = y_part_values[k];
      y_col_indices[x_batch_pointers[j] + k] = y_part_col_indices[k];
    }
  }

  if (conjugate == false) {
  }
  auto output = ctx.Output(2);
  auto output_shape = output->GetTensorShape();
  if (rank == ktwo) {
    output_shape->SetDimSizes({num_cols + 1});
  } else {
    output_shape->SetDimSizes({x_dense_shape[0] * (num_cols + 1)});
  }
  output->SetTensorShape(output_shape.get());
  return KERNEL_STATUS_OK;
}

template <typename indiceT, typename valueT>
uint32_t SparseMatrixTransposeCpuKernel::SparseMatrixTransposeComputecomplex(CpuKernelContext &ctx) {
  indiceT *x_dense_shape = static_cast<indiceT *>(ctx.Input(0)->GetData());
  indiceT *x_batch_pointers = static_cast<indiceT *>(ctx.Input(1)->GetData());
  indiceT *x_row_pointers = static_cast<indiceT *>(ctx.Input(2)->GetData());
  indiceT *x_col_indices = static_cast<indiceT *>(ctx.Input(3)->GetData());
  valueT *x_values = static_cast<valueT *>(ctx.Input(4)->GetData());
  bool conjugate = (ctx.GetAttr("conjugate") == nullptr) ? false : ctx.GetAttr("conjugate")->GetBool();
  indiceT *y_dense_shape = static_cast<indiceT *>(ctx.Output(0)->GetData());
  indiceT *y_batch_pointers = static_cast<indiceT *>(ctx.Output(1)->GetData());
  indiceT *y_row_pointers = static_cast<indiceT *>(ctx.Output(2)->GetData());
  indiceT *y_col_indices = static_cast<indiceT *>(ctx.Output(3)->GetData());
  valueT *y_values = static_cast<valueT *>(ctx.Output(4)->GetData());
  auto rank = ctx.Input(0)->NumElements();
  if (rank == krankwithbatch) {
    y_dense_shape[0] = x_dense_shape[0];
    y_dense_shape[1] = x_dense_shape[ktwo];
    y_dense_shape[ktwo] = x_dense_shape[1];
  } else {
    y_dense_shape[0] = x_dense_shape[1];
    y_dense_shape[1] = x_dense_shape[0];
  }
  auto batch_pointers = ctx.Input(1)->NumElements();
  for (int i = 0; i < batch_pointers; ++i) {
    y_batch_pointers[i] = x_batch_pointers[i];
  }

  auto num_rows = x_dense_shape[rank - 2];
  auto num_cols = x_dense_shape[rank - 1];
  auto num_batch = ctx.Input(1)->NumElements() - 1;
  int y_part_row_pointers[num_cols + 1];
  int part_row_pointers[num_rows + 1];

  for (int j = 0; j < num_batch; ++j) {
    int n = x_batch_pointers[j + 1] - x_batch_pointers[j];
    valueT part_values[n];
    indiceT part_col_indices[n];
    indiceT y_part_col_indices[n];
    valueT y_part_values[n];
    for (int i = 0; i < num_cols + 1; ++i) {
      y_part_row_pointers[i] = 0;
    }
    for (int k = 0; k < num_rows + 1; ++k) {
      part_row_pointers[k] = x_row_pointers[(num_rows + 1) * j + k];
    }
    for (int k = 0; k < n; ++k) {
      part_values[k] = x_values[x_batch_pointers[j] + k];
      part_col_indices[k] = x_col_indices[x_batch_pointers[j] + k];
    }
    for (int64_t i = 0; i < n; ++i) {
      y_part_row_pointers[part_col_indices[i] + 1] += 1;
    }
    std::partial_sum(y_part_row_pointers, y_part_row_pointers + num_cols + 1, y_part_row_pointers);
    for (int k = 0; k < num_cols + 1; ++k) {
      y_row_pointers[(num_cols + 1) * j + k] = y_part_row_pointers[k];
    }

    for (int k = 0; k < n; ++k) {
      part_values[k] = x_values[x_batch_pointers[j] + k];
      part_col_indices[k] = x_col_indices[x_batch_pointers[j] + k];
    }
    std::vector<int> current_col_count(num_cols);
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const int64_t row_begin = part_row_pointers[row_idx];
      const int64_t row_end = part_row_pointers[row_idx + 1];
      for (int64_t i = row_begin; i < row_end; ++i) {
        const int col_idx = part_col_indices[i];
        const int64_t offset = y_part_row_pointers[col_idx] + current_col_count[col_idx];
        y_part_col_indices[offset] = row_idx;
        y_part_values[offset] = part_values[i];
        current_col_count[col_idx] += 1;
      }
    }
    for (int k = 0; k < n; ++k) {
      y_values[x_batch_pointers[j] + k] = y_part_values[k];
      y_col_indices[x_batch_pointers[j] + k] = y_part_col_indices[k];
    }
  }

  if (conjugate == true) {
    for (int i = 0; i < ctx.Input(kfour)->GetTensorShape()->NumElements(); ++i) {
      y_values[i] = std::conj(y_values[i]);
    }
  }
  auto output = ctx.Output(2);
  auto output_shape = output->GetTensorShape();
  if (rank == ktwo) {
    output_shape->SetDimSizes({num_cols + 1});
  } else {
    output_shape->SetDimSizes({x_dense_shape[0] * (num_cols + 1)});
  }
  output->SetTensorShape(output_shape.get());
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(SPARSEMATRIXTRANSPOSE, SparseMatrixTransposeCpuKernel);
}  // namespace aicpu
