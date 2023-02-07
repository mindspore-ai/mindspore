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
#include "sparse_tensor_to_csr_sparse_matrix.h"

#include <complex>
#include <numeric>

#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 5;
const char *SparseTensorToCSRSparseMatrix = "SparseTensorToCSRSparseMatrix";
const int DIM2 = 2;
const int DIM3 = 3;
}  // namespace

namespace aicpu {
uint32_t SparseTensorToCSRSparseMatrixCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SparseTensorToCSRSparseMatrix normal check failed.");
  Tensor *x_indices = ctx.Input(0);
  Tensor *x_values = ctx.Input(1);
  Tensor *x_dense_shape = ctx.Input(2);

  const int rank = x_dense_shape->NumElements();
  if (rank != DIM2 && rank != DIM3) {
    KERNEL_LOG_ERROR("SparseTensor must have rank 2 or 3.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_indices_shape = x_indices->GetTensorShape();
  auto x_values_shape = x_values->GetTensorShape();
  if (x_indices_shape->NumElements() / rank != x_values_shape->NumElements()) {
    KERNEL_LOG_ERROR("Tensor x_indices&x_values's ranks mismatch.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_dense_shape_data_type = x_dense_shape->GetDataType();
  auto x_indices_data_type = x_indices->GetDataType();
  if (x_indices_data_type != DT_INT32 && x_indices_data_type != DT_INT64) {
    KERNEL_LOG_ERROR("SparseTensorToCSRSparseMatrix kernel data type [%s] not support.",
                     DTypeStr(x_indices_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (x_dense_shape_data_type != x_indices_data_type) {
    KERNEL_LOG_ERROR("SparseTensorToCSRSparseMatrix kernel data type mismatch.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_values_data_type = x_values->GetDataType();

  uint32_t status;
  switch (x_indices_data_type) {
    case DT_INT32:
      switch (x_values_data_type) {
        case DT_FLOAT:
          status = ComputeKernel<int32_t, float>(ctx);
          break;
        case DT_DOUBLE:
          status = ComputeKernel<int32_t, double>(ctx);
          break;
        case DT_COMPLEX64:
          status = ComputeKernel<int32_t, std::complex<float> >(ctx);
          break;
        case DT_COMPLEX128:
          status = ComputeKernel<int32_t, std::complex<double> >(ctx);
          break;
        default:
          KERNEL_LOG_ERROR(
            "SparseTensorToCSRSparseMatrix kernel data type [%s] not "
            "support.",
            DTypeStr(x_values_data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (x_values_data_type) {
        case DT_FLOAT:
          status = ComputeKernel<int64_t, float>(ctx);
          break;
        case DT_DOUBLE:
          status = ComputeKernel<int64_t, double>(ctx);
          break;
        case DT_COMPLEX64:
          status = ComputeKernel<int64_t, std::complex<float> >(ctx);
          break;
        case DT_COMPLEX128:
          status = ComputeKernel<int64_t, std::complex<double> >(ctx);
          break;
        default:
          KERNEL_LOG_ERROR(
            "SparseTensorToCSRSparseMatrix kernel data type [%s] not "
            "support.",
            DTypeStr(x_values_data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      KERNEL_LOG_ERROR("data type of indices is not int32 or int64");
      return KERNEL_STATUS_PARAM_INVALID;
  }

  if (status != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("SparseTensorToCSRSparseMatrix kernel compute failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(SparseTensorToCSRSparseMatrix, SparseTensorToCSRSparseMatrixCpuKernel);

template <typename indicesT, typename dataT>
uint32_t SparseTensorToCSRSparseMatrixCpuKernel::ComputeKernel(CpuKernelContext &ctx) {
  auto x_dense_shape = ctx.Input(2);
  auto x_dense_shape_ptr = static_cast<indicesT *>(x_dense_shape->GetData());
  auto y_dense_shape_ptr = static_cast<indicesT *>(ctx.Output(0)->GetData());
  auto x_values_ptr = static_cast<dataT *>(ctx.Input(1)->GetData());
  auto y_values_ptr = static_cast<dataT *>(ctx.Output(4)->GetData());

  // Copy the CSRSparseMatrix's dense_shape and values from the SparseTensor.
  for (int64_t i = 0; i < x_dense_shape->GetTensorShape()->NumElements(); i++) {
    y_dense_shape_ptr[i] = x_dense_shape_ptr[i];
  }
  for (int64_t i = 0; i < ctx.Input(1)->GetTensorShape()->NumElements(); i++) {
    y_values_ptr[i] = x_values_ptr[i];
  }

  auto y_batch_pointers_ptr = static_cast<indicesT *>(ctx.Output(1)->GetData());
  auto y_row_pointers_ptr = static_cast<indicesT *>(ctx.Output(2)->GetData());
  auto y_col_indices_ptr = static_cast<indicesT *>(ctx.Output(3)->GetData());
  auto x_indices_ptr = static_cast<indicesT *>(ctx.Input(0)->GetData());

  const int rank = ctx.Input(2)->NumElements();
  const int64_t batch_size = (rank == DIM2) ? 1 : x_dense_shape_ptr[0];
  const int64_t num_rows = x_dense_shape_ptr[(rank == DIM2) ? 0 : 1];
  const int64_t total_nnz = ctx.Input(1)->NumElements();

  for (int64_t i = 0; i < batch_size * (num_rows + 1); i++) {
    y_row_pointers_ptr[i] = 0;
  }

  int64_t prev_batch = -1;
  if (rank == DIM2) {
    // For a single batch, the batch_ptrs are {0, total_nnz}.
    y_batch_pointers_ptr[0] = 0;
    ++prev_batch;

    for (int64_t i = 0; i < total_nnz; ++i) {
      // For now, the rows pointers store the corresponding row counts.
      int64_t offset = i * rank;
      y_row_pointers_ptr[x_indices_ptr[offset] + 1] += 1;
      y_col_indices_ptr[i] = x_indices_ptr[++offset];
    }
  } else {  // rank == 3
    for (int64_t i = 0; i < total_nnz; ++i) {
      int64_t offset = i * rank;
      const int cur_batch = x_indices_ptr[offset];
      // For now, the rows pointers store the corresponding row counts.
      y_row_pointers_ptr[cur_batch * (num_rows + 1) + x_indices_ptr[++offset] + 1] += 1;
      y_col_indices_ptr[i] = x_indices_ptr[++offset];

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
