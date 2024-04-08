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
#include "cpu_kernel/ms_kernel/csr_sparse_matrix_to_sparse_tensor.h"

#include <algorithm>
#include <complex>
#include <iostream>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 5;
const uint32_t kOutputNum = 3;
const char *CSRSparseMatrixToSparseTensor = "CSRSparseMatrixToSparseTensor";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 4;
const int64_t kParallelDataNumMid = 32;
const int DIM2 = 2;
const int DIM3 = 3;
}  // namespace

namespace aicpu {
uint32_t CSRSparseMatrixToSparseTensorCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "CSRSparseMatrixToSparseTensor normal check failed.");
  Tensor *x_dense_shape = ctx.Input(0);
  Tensor *x_batch_pointers = ctx.Input(1);
  Tensor *x_row_pointers = ctx.Input(2);
  Tensor *x_col_indices = ctx.Input(3);
  Tensor *x_values = ctx.Input(4);

  const int rank = x_dense_shape->NumElements();
  if (rank != DIM2 && rank != DIM3) {
    CUST_KERNEL_LOG_ERROR(ctx, "CSR SparseMatrix must have rank 2 or 3.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_row_pointers_shape = x_row_pointers->GetTensorShape();
  auto x_col_indices_shape = x_col_indices->GetTensorShape();
  auto x_values_shape = x_values->GetTensorShape();
  if (x_col_indices_shape->NumElements() != x_values_shape->NumElements()) {
    CUST_KERNEL_LOG_ERROR(ctx, "Tensor x_col_indices&x_values's ranks mismatch.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_dense_shape_data_type = x_dense_shape->GetDataType();
  auto x_batch_pointers_data_type = x_batch_pointers->GetDataType();
  auto x_row_pointers_data_type = x_row_pointers->GetDataType();
  auto x_col_indices_data_type = x_col_indices->GetDataType();
  if (x_col_indices_data_type != DT_INT32 && x_col_indices_data_type != DT_INT64) {
    CUST_KERNEL_LOG_ERROR(ctx, "CSRSparseMatrixToSparseTensor kernel data type [%s] not support.",
                          DTypeStr(x_col_indices_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (x_dense_shape_data_type != x_col_indices_data_type || x_batch_pointers_data_type != x_col_indices_data_type ||
      x_row_pointers_data_type != x_col_indices_data_type) {
    CUST_KERNEL_LOG_ERROR(ctx, "CSRSparseMatrixToSparseTensor kernel data type mismatch.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_values_data_type = x_values->GetDataType();

  uint32_t status;
  switch (x_col_indices_data_type) {
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
          CUST_KERNEL_LOG_ERROR(ctx,
                                "CSRSparseMatrixToSparseTensor kernel data type [%s] not "
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
          CUST_KERNEL_LOG_ERROR(ctx,
                                "CSRSparseMatrixToSparseTensor kernel data type [%s] not "
                                "support.",
                                DTypeStr(x_values_data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "data type of indices is not int32 or int64");
      return KERNEL_STATUS_PARAM_INVALID;
  }

  if (status != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "CSRSparseMatrixToSparseTensor kernel compute failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(CSRSparseMatrixToSparseTensor, CSRSparseMatrixToSparseTensorCpuKernel);

template <typename indicesT, typename dataT>
uint32_t CSRSparseMatrixToSparseTensorCpuKernel::ComputeKernel(CpuKernelContext &ctx) {
  auto x_dense_shape = ctx.Input(0);
  auto x_dense_shape_ptr = static_cast<indicesT *>(x_dense_shape->GetData());
  auto dense_shape_ptr = static_cast<indicesT *>(ctx.Output(2)->GetData());
  auto values_ptr = static_cast<dataT *>(ctx.Output(1)->GetData());
  auto x_values = ctx.Input(4);
  auto x_values_ptr = static_cast<dataT *>(x_values->GetData());

  // Copy the SparseTensor's dense_shape and values from the CSRSparseMatrix.
  for (int64_t i = 0; i < x_dense_shape->GetTensorShape()->NumElements(); i++) {
    dense_shape_ptr[i] = x_dense_shape_ptr[i];
  }
  for (int64_t i = 0; i < x_values->GetTensorShape()->NumElements(); i++) {
    values_ptr[i] = x_values_ptr[i];
  }

  const uint32_t batch_size = ctx.Input(1)->NumElements() - 1;
  if (batch_size >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (batch_size <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > batch_size) {
      max_core_num = batch_size;
    }

    auto sharder = [&](int64_t batch_begin, int64_t batch_end) {
      SpecialCompute<indicesT>(batch_begin, batch_end, ctx);
    };

    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
    }

    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, batch_size, batch_size / max_core_num, sharder),
                             "CSRSparseMatrixToSparseTensor Compute failed.");
  } else {
    SpecialCompute<indicesT>(0, batch_size, ctx);
  }

  return KERNEL_STATUS_OK;
}

template <typename indicesT>
void CSRSparseMatrixToSparseTensorCpuKernel::SpecialCompute(int64_t batch_begin, int64_t batch_end,
                                                            CpuKernelContext &ctx) {
  auto x_dense_shape = ctx.Input(0);
  const int rank = x_dense_shape->NumElements();
  auto x_dense_shape_ptr = static_cast<indicesT *>(x_dense_shape->GetData());
  const int64_t num_rows = x_dense_shape_ptr[(rank == DIM2) ? 0 : 1];
  auto x_batch_pointers_ptr = static_cast<indicesT *>(ctx.Input(1)->GetData());
  auto x_row_pointers_ptr = static_cast<indicesT *>(ctx.Input(2)->GetData());
  auto x_col_indices_ptr = static_cast<indicesT *>(ctx.Input(3)->GetData());

  for (int64_t batch_idx = batch_begin; batch_idx < batch_end; ++batch_idx) {
    const int64_t batch_offset = x_batch_pointers_ptr[batch_idx];

    for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
      int64_t row_offset = batch_idx * (num_rows + 1) + row_idx;

      // The column indices of the current row lie in the range:
      //  [x_row_pointers_ptr[row_offset], x_row_pointer_ptr[row_offset + 1]]
      const int64_t col_begin = x_row_pointers_ptr[row_offset];
      const int64_t col_end = x_row_pointers_ptr[row_offset + 1];
      for (int64_t i = col_begin; i < col_end; ++i) {
        const int64_t col_idx = x_col_indices_ptr[batch_offset + i];
        const int64_t indices_offset = rank * (batch_offset + i);
        IndicesCompute<indicesT>(ctx, indices_offset, batch_idx, row_idx, col_idx);
      }
    }
  }
}

template <typename indicesT>
void CSRSparseMatrixToSparseTensorCpuKernel::IndicesCompute(CpuKernelContext &ctx, int64_t indices_offset,
                                                            const int64_t batch_idx, const int64_t row_idx,
                                                            const int64_t col_idx) {
  const int rank = ctx.Input(0)->NumElements();
  auto indices_ptr = static_cast<indicesT *>(ctx.Output(0)->GetData());
  if (rank == DIM2) {
    indices_ptr[indices_offset] = row_idx;
    indices_ptr[++indices_offset] = col_idx;
  } else {  // rank == 3
    indices_ptr[indices_offset] = batch_idx;
    indices_ptr[++indices_offset] = row_idx;
    indices_ptr[++indices_offset] = col_idx;
  }
}
}  // namespace aicpu
