/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cpu_kernel/ms_kernel/dense_to_csr_sparse_matrix.h"

#include <complex>
#include <numeric>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 5;
const char *DenseToCSRSparseMatrix = "DenseToCSRSparseMatrix";
}  // namespace

namespace aicpu {
uint32_t DenseToCSRSparseMatrixCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "DenseToCSRSparseMatrix normal check failed.");
  DataType value_type = ctx.Input(0)->GetDataType();
  DataType indice_type = ctx.Input(1)->GetDataType();
  uint32_t status;
  switch (indice_type) {
    case DT_INT32:
      switch (value_type) {
        case DT_FLOAT:
          status = ComputeKernel<int32_t, float>(ctx);
          break;
        case DT_DOUBLE:
          status = ComputeKernel<int32_t, double>(ctx);
          break;
        case DT_COMPLEX64:
          status = ComputeKernel<int32_t, std::complex<float>>(ctx);
          break;
        case DT_COMPLEX128:
          status = ComputeKernel<int32_t, std::complex<double>>(ctx);
          break;
        default:
          KERNEL_LOG_ERROR("DenseToCSRSparseMatrix value type [%s] not support.", DTypeStr(value_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (value_type) {
        case DT_FLOAT:
          status = ComputeKernel<int64_t, float>(ctx);
          break;
        case DT_DOUBLE:
          status = ComputeKernel<int64_t, double>(ctx);
          break;
        case DT_COMPLEX64:
          status = ComputeKernel<int64_t, std::complex<float>>(ctx);
          break;
        case DT_COMPLEX128:
          status = ComputeKernel<int64_t, std::complex<double>>(ctx);
          break;
        default:
          KERNEL_LOG_ERROR("DenseToCSRSparseMatrix value type [%s] not support.", DTypeStr(value_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      KERNEL_LOG_ERROR("DenseToCSRSparseMatrix indices type [%s] not support.", DTypeStr(indice_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_HANDLE_ERROR(status, "DenseToCSRSparseMatrix kernel compute failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(DenseToCSRSparseMatrix, DenseToCSRSparseMatrixCpuKernel);

template <typename indiceT, typename valueT>
uint32_t DenseToCSRSparseMatrixCpuKernel::ComputeKernel(const CpuKernelContext &ctx) {
  auto dense_input_ptr = reinterpret_cast<valueT *>(ctx.Input(0)->GetData());
  auto indices_ptr = reinterpret_cast<indiceT *>(ctx.Input(1)->GetData());
  auto y_dense_shape_ptr = reinterpret_cast<indiceT *>(ctx.Output(0)->GetData());
  auto y_batch_pointers_ptr = reinterpret_cast<indiceT *>(ctx.Output(1)->GetData());
  auto y_row_pointers_ptr = reinterpret_cast<indiceT *>(ctx.Output(2)->GetData());
  auto y_col_indices_ptr = reinterpret_cast<indiceT *>(ctx.Output(3)->GetData());
  auto y_values_ptr = reinterpret_cast<valueT *>(ctx.Output(4)->GetData());
  // Copy the CSRSparseMatrix's dense_shape and values from the Dense.
  const int64_t rank = ctx.Input(1)->GetTensorShape()->GetDimSize(1);
  const int64_t total_nnz = ctx.Input(1)->GetTensorShape()->GetDimSize(0);
  const int64_t batch_size = (rank == 2) ? 1 : ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  const int64_t num_rows = ctx.Input(0)->GetTensorShape()->GetDimSize((rank == 2) ? 0 : 1);
  const int64_t num_cols = ctx.Input(0)->GetTensorShape()->GetDimSize((rank == 2) ? 1 : 2);
  for (int64_t i = 0; i < rank; i++) {
    y_dense_shape_ptr[i] = ctx.Input(0)->GetTensorShape()->GetDimSize(i);
  }
  for (int64_t i = 0; i < total_nnz; i++) {
    if (rank == 2) {
      int64_t cur_idx = indices_ptr[i * rank] * num_cols + indices_ptr[i * rank + 1];
      y_values_ptr[i] = dense_input_ptr[cur_idx];
    } else {
      int64_t cur_idx = indices_ptr[i * rank] * num_rows * num_cols;
      cur_idx = cur_idx + indices_ptr[i * rank + 1] * num_cols + indices_ptr[i * rank + 2];
      y_values_ptr[i] = dense_input_ptr[cur_idx];
    }
  }
  for (int64_t i = 0; i < batch_size * (num_rows + 1); i++) {
    y_row_pointers_ptr[i] = 0;
  }
  int prev_batch = -1;
  if (rank == 2) {
    // For a single batch, the batch_ptrs are {0, total_nnz}.
    y_batch_pointers_ptr[0] = 0;
    ++prev_batch;
    for (int64_t i = 0; i < total_nnz; ++i) {
      // For now, the rows pointers store the corresponding row counts.
      y_row_pointers_ptr[indices_ptr[i * rank] + 1] += 1;
      y_col_indices_ptr[i] = indices_ptr[i * rank + 1];
    }
  } else {  // rank == 3
    for (int64_t i = 0; i < total_nnz; ++i) {
      const int cur_batch = indices_ptr[i * rank];
      // For now, the rows pointers store the corresponding row counts.
      y_row_pointers_ptr[cur_batch * (num_rows + 1) + indices_ptr[i * rank + 1] + 1] += 1;
      y_col_indices_ptr[i] = indices_ptr[i * rank + 2];
      // We're at a new batch and might have skipped over empty batches.
      while (prev_batch < cur_batch) {
        // The previous batch ends at position i.
        y_batch_pointers_ptr[prev_batch + 1] = i;
        ++prev_batch;
      }
    }
  }
  // Set the last element of batch_ptr and account for trailing empty batches.
  while (prev_batch < batch_size) {
    y_batch_pointers_ptr[prev_batch + 1] = total_nnz;
    ++prev_batch;
  }
  // Compute the cumulative row counts for each batch.
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto *row_ptr_batch = y_row_pointers_ptr + batch_idx * (num_rows + 1);
    std::partial_sum(row_ptr_batch, row_ptr_batch + num_rows + 1, row_ptr_batch);
  }
  return KERNEL_STATUS_OK;
}
}  // namespace aicpu
