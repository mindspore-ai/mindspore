/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All right reserved.
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

#include "tril_indices.h"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <map>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kTrilIndices = "TrilIndices";

#define TRIL_INDICES_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                             \
    uint32_t result = DoCompute<TYPE>(CTX);                   \
    if (result != KERNEL_STATUS_OK) {                         \
      KERNEL_LOG_ERROR("TrilIndices kernel compute failed."); \
      return result;                                          \
    }                                                         \
    break;                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t TrilIndicesCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_PARAM_INVALID, "Get output failed.")
  auto data_type = ctx.Output(0)->GetDataType();
  switch (data_type) {
    TRIL_INDICES_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TRIL_INDICES_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    default:
      KERNEL_LOG_ERROR("TrilIndices kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TrilIndicesCpuKernel::DoCompute(CpuKernelContext &ctx) {
  AttrValue *row_ptr = ctx.GetAttr("row");
  AttrValue *col_ptr = ctx.GetAttr("col");
  AttrValue *offset_ptr = ctx.GetAttr("offset");
  int64_t row = row_ptr->GetInt();
  int64_t col = col_ptr->GetInt();
  int64_t offset = (offset_ptr == nullptr) ? 0 : (offset_ptr->GetInt());

  auto m_first_row = offset > 0 ? std::min<int64_t>(col, 1 + offset) : row + offset > 0;
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);
  auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }

  T *output{static_cast<T *>(ctx.Output(0)->GetData())};

  int64_t i = 0;
  int64_t r = std::max<int64_t>(0, -offset), c = 0;
  while (i < tril_size) {
    output[i] = r;
    output[tril_size + i++] = c;
    c += 1;
    if (c > r + offset || c >= col) {
      r += 1;
      c = 0;
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTrilIndices, TrilIndicesCpuKernel);
}  // namespace aicpu
