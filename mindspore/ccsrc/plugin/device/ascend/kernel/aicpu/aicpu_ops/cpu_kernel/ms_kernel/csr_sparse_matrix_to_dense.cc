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

#include "cpu_kernel/ms_kernel/csr_sparse_matrix_to_dense.h"

#include <securec.h>
#include <algorithm>
#include <complex>
#include <numeric>
#include <string>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "common/kernel_log.h"
#include "common/status.h"
#include "utils/allocator_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 5;
const uint32_t kOutputNum = 1;
const char *CSRSparseMatrixToDense = "CSRSparseMatrixToDense";

#define SWITCH_CASE(_IDX_T, _VALUE_T, VALUE_T, FLAG, CTX)                                                  \
  case _VALUE_T:                                                                                           \
    switch (_IDX_T) {                                                                                      \
      case DT_INT32:                                                                                       \
        (FLAG) = DoCompute<int32_t, VALUE_T>(CTX);                                                         \
        break;                                                                                             \
      case DT_INT64:                                                                                       \
        (FLAG) = DoCompute<int64_t, VALUE_T>(CTX);                                                         \
        break;                                                                                             \
      default:                                                                                             \
        KERNEL_LOG_ERROR("CSRSparseMatrixToDense index type [%s] not support.", DTypeStr(_IDX_T).c_str()); \
        return KERNEL_STATUS_PARAM_INVALID;                                                                \
    }                                                                                                      \
    break;

}  // namespace

namespace aicpu {
uint32_t CSRSparseMatrixToDenseCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "CSRSparseMatrixToDense normal check failed.");
  DataType indice_type = ctx.Input(0)->GetDataType();
  DataType value_type = ctx.Input(4)->GetDataType();
  uint32_t status = 0;
  switch (value_type) {
    SWITCH_CASE(indice_type, DT_FLOAT, float_t, status, ctx)
    SWITCH_CASE(indice_type, DT_DOUBLE, double_t, status, ctx)
    SWITCH_CASE(indice_type, DT_COMPLEX64, std::complex<float_t>, status, ctx)
    SWITCH_CASE(indice_type, DT_COMPLEX128, std::complex<double_t>, status, ctx)
    default:
      KERNEL_LOG_ERROR("CSRSparseMatrixToDense values type [%s] not support.", DTypeStr(value_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_HANDLE_ERROR(status, "CSRSparseMatrixToDense compute failed!");
  return KERNEL_STATUS_OK;
}

template <typename indiceT, typename valueT>
uint32_t CSRSparseMatrixToDenseCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  indiceT batch_size = ctx.Input(1)->NumElements() - 1;
  auto rank = ctx.Input(0)->NumElements();
  int shift = (rank == 2) ? 0 : 1;
  indiceT num_rows = *(static_cast<indiceT *>(ctx.Input(0)->GetData()) + shift);
  indiceT num_cols = *(static_cast<indiceT *>(ctx.Input(0)->GetData()) + shift + 1);
  indiceT *batch_ptrs = static_cast<indiceT *>(ctx.Input(1)->GetData());
  indiceT *row_ptrs = static_cast<indiceT *>(ctx.Input(2)->GetData());
  indiceT *col_ind = static_cast<indiceT *>(ctx.Input(3)->GetData());
  valueT *values = static_cast<valueT *>(ctx.Input(4)->GetData());
  auto output = ctx.Output(0);
  auto output_shape = output->GetTensorShape();
  if (rank == 2) {
    output_shape->SetDimSizes({num_rows, num_cols});
  } else {
    output_shape->SetDimSizes({batch_size, num_rows, num_cols});
  }
  output->SetTensorShape(output_shape.get());
  valueT *y_data = static_cast<valueT *>(ctx.Output(0)->GetData());
  // use multi-thread
  uint32_t min_core = 1;
  uint32_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  max_core = std::min(max_core, (uint32_t)batch_size);
  if (max_core == 0) {
    KERNEL_LOG_ERROR("Max core num cannot be zero.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto shard = [&](int64_t start, int64_t end) {
    for (int64_t batch_idx = start; batch_idx < end; batch_idx++) {
      const int64_t dense_offset = batch_idx * num_rows * num_cols;
      for (int64_t i = 0; i < num_rows * num_cols; ++i) {
        y_data[dense_offset + i] = 0;
      }
      const int64_t csr_batch_offset = batch_ptrs[batch_idx];
      for (int64_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const int64_t row_offset = batch_idx * (num_rows + 1) + row_idx;
        const int64_t col_begin = row_ptrs[row_offset];
        const int64_t col_end = row_ptrs[row_offset + 1];
        for (int64_t i = col_begin; i < col_end; ++i) {
          const int64_t col_idx = col_ind[csr_batch_offset + i];
          y_data[dense_offset + (row_idx * num_cols) + col_idx] = values[csr_batch_offset + i];
        }
      }
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch_size, batch_size / max_core, shard),
                      "CSRSparseMatrixToDense Compute failed.");
  return KERNEL_STATUS_OK;
}

// register the opetaor
REGISTER_CPU_KERNEL(CSRSparseMatrixToDense, CSRSparseMatrixToDenseCpuKernel);
}  // namespace aicpu
