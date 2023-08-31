/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/matrix_band_part.h"

#include <securec.h>
#include <algorithm>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "common/kernel_log.h"
#include "cpu_kernel/common/status.h"

namespace {
const char *kMatrixBandPart = "MatrixBandPart";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
constexpr int64_t kParallelDataNums = 64 * 1024;

#define BAND_COMPUTE_CASE(DTYPE, TYPE, X, LOWER, UPPER, Y, M, N, CTX)   \
  case (DTYPE): {                                                       \
    uint32_t result = BandCompute<TYPE>(X, LOWER, UPPER, Y, M, N, CTX); \
    if (result != KERNEL_STATUS_OK) {                                   \
      KERNEL_LOG_ERROR("MatrixBandPart kernel compute failed.");        \
      return result;                                                    \
    }                                                                   \
    break;                                                              \
  }
}  // namespace

namespace aicpu {
uint32_t MatrixBandPartCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MatrixBandPart check input and output number failed.");
  Tensor *x = ctx.Input(0);
  Tensor *num_lower = ctx.Input(1);
  Tensor *num_upper = ctx.Input(2);
  Tensor *y = ctx.Output(0);
  int32_t rank = x->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE((rank >= 2), KERNEL_STATUS_PARAM_INVALID, "Input must be at least 2-dim, but get dims: %d", rank);
  int64_t m = x->GetTensorShape()->GetDimSize(rank - 2);
  int64_t n = x->GetTensorShape()->GetDimSize(rank - 1);
  auto lower_shape = num_lower->GetTensorShape();
  auto upper_shape = num_upper->GetTensorShape();
  KERNEL_CHECK_FALSE(
    (lower_shape->GetDimSizes().empty() || (lower_shape->GetDims() == 1 && lower_shape->GetDimSize(0) == 1)),
    KERNEL_STATUS_PARAM_INVALID, "num_lower must be scalar or a single 1-dimension number.");
  KERNEL_CHECK_FALSE(
    (upper_shape->GetDimSizes().empty() || (upper_shape->GetDims() == 1 && upper_shape->GetDimSize(0) == 1)),
    KERNEL_STATUS_PARAM_INVALID, "num_upper must be scalar or a single 1-dimension number.");
  DataType lower_type = num_lower->GetDataType();
  KERNEL_CHECK_FALSE((lower_type == DT_INT32 || lower_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "Unsupported num_lower data_type[%s], "
                     "only support DT_INT32 and DT_INT64.",
                     DTypeStr(lower_type).c_str());
  DataType upper_type = num_upper->GetDataType();
  KERNEL_CHECK_FALSE((upper_type == DT_INT32 || upper_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                     "Unsupported num_upper data_type[%s], "
                     "only support DT_INT32 and DT_INT64.",
                     DTypeStr(upper_type).c_str());
  int32_t *lower_data = reinterpret_cast<int32_t *>(num_lower->GetData());
  KERNEL_CHECK_NULLPTR(lower_data, KERNEL_STATUS_PARAM_INVALID, "Get num_lower data failed.");
  int32_t *upper_data = reinterpret_cast<int32_t *>(num_upper->GetData());
  KERNEL_CHECK_NULLPTR(upper_data, KERNEL_STATUS_PARAM_INVALID, "Get num_upper data failed.");
  int64_t lower = *lower_data;
  int64_t upper = *upper_data;
  KERNEL_CHECK_FALSE((lower <= m), KERNEL_STATUS_PARAM_INVALID,
                     "num_lower must be negative or less or equal to number "
                     "of rows [%d], got: [%d]",
                     m, lower);
  KERNEL_CHECK_FALSE((upper <= n), KERNEL_STATUS_PARAM_INVALID,
                     "num_lower must be negative or less or equal to number "
                     "of cols [%d], got: [%d]",
                     n, upper);
  uint64_t input_size = x->GetDataSize();
  uint64_t output_size = y->GetDataSize();
  KERNEL_CHECK_FALSE((input_size == output_size), KERNEL_STATUS_PARAM_INVALID,
                     "Input data size[%llu] is not equal to output data size[%llu].", input_size, output_size);
  DataType data_type = x->GetDataType();
  switch (data_type) {
    BAND_COMPUTE_CASE(DT_INT8, int8_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_INT16, int16_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_INT32, int32_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_INT64, int64_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_UINT8, uint8_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_UINT16, uint16_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_UINT32, uint32_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_UINT64, uint64_t, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_FLOAT16, Eigen::half, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_FLOAT, float, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_DOUBLE, double, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_BOOL, bool, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, x, lower, upper, y, m, n, ctx)
    BAND_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, x, lower, upper, y, m, n, ctx)
    default:
      KERNEL_LOG_ERROR("MatrixBandPart kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixBandPartCpuKernel::BandCompute(Tensor *x, int64_t lower, int64_t upper, Tensor *y, int64_t rows,
                                              int64_t cols, const CpuKernelContext &ctx) {
  T *x_addrs = reinterpret_cast<T *>(x->GetData());
  KERNEL_CHECK_NULLPTR(x_addrs, KERNEL_STATUS_PARAM_INVALID, "Get input data failed.");
  T *y_addrs = reinterpret_cast<T *>(y->GetData());
  KERNEL_CHECK_NULLPTR(y_addrs, KERNEL_STATUS_PARAM_INVALID, "Get output data failed.");

  T zero = static_cast<T>(0);
  int64_t data_num = x->GetDataSize() / sizeof(T);
  int64_t matrix_size = rows * cols;
  int64_t total_rows = data_num / cols;
  bool same_addr = (x_addrs == y_addrs);
  if (data_num < kParallelDataNums) {
    if (!same_addr) {
      std::fill_n(y_addrs, data_num, zero);
    }
    int64_t batch_end = (total_rows + rows - 1) / rows;
    for (int64_t i = 0; i < batch_end; i++) {
      int64_t row_begin = 0 > i * rows ? 0 % rows : 0;
      int64_t row_end = total_rows < (i + 1) * rows ? total_rows % rows : rows;
      for (int64_t m = row_begin; m < row_end; m++) {
        int64_t base_index = i * matrix_size + m * cols;
        int64_t band_start = lower < 0 ? 0 : std::min(cols, std::max(int64_t{0}, int64_t{m - lower}));
        int64_t band_end = upper < 0 ? cols : std::min(cols, int64_t{m + upper + 1});
        if (same_addr) {
          if (band_start > 0) {
            std::fill((y_addrs + base_index), (y_addrs + base_index + band_start), zero);
          }
          if (band_end < cols) {
            std::fill((y_addrs + base_index + band_end), (y_addrs + base_index + cols), zero);
          }
        } else {
          if (band_start < band_end) {
            (void)memcpy_s((y_addrs + base_index + band_start), (band_end - band_start) * sizeof(T),
                           (x_addrs + base_index + band_start), (band_end - band_start) * sizeof(T));
          }
        }
      }
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > total_rows) {
      max_core_num = total_rows;
    }
    auto shard_band = [&](int64_t start, int64_t end) {
      if (!same_addr) {
        std::fill(y_addrs + start * cols, y_addrs + end * cols, zero);
      }
      int64_t batch_begin = start / rows;
      int64_t batch_end = (end + rows - 1) / rows;
      for (int64_t i = batch_begin; i < batch_end; i++) {
        int64_t row_begin = start > i * rows ? start % rows : 0;
        int64_t row_end = end < (i + 1) * rows ? end % rows : rows;
        for (int64_t m = row_begin; m < row_end; m++) {
          int64_t base_index = i * matrix_size + m * cols;
          int64_t band_start = lower < 0 ? 0 : std::min(cols, std::max(int64_t{0}, int64_t{m - lower}));
          int64_t band_end = upper < 0 ? cols : std::min(cols, int64_t{m + upper + 1});
          if (same_addr) {
            if (band_start > 0) {
              std::fill((y_addrs + base_index), (y_addrs + base_index + band_start), zero);
            }
            if (band_end < cols) {
              std::fill((y_addrs + base_index + band_end), (y_addrs + base_index + cols), zero);
            }
          } else {
            if (band_start < band_end) {
              (void)memcpy_s((y_addrs + base_index + band_start), (band_end - band_start) * sizeof(T),
                             (x_addrs + base_index + band_start), (band_end - band_start) * sizeof(T));
            }
          }
        }
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, total_rows, total_rows / max_core_num, shard_band),
                        "MatrixBandPart Compute failed.");
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMatrixBandPart, MatrixBandPartCpuKernel);
}  // namespace aicpu
