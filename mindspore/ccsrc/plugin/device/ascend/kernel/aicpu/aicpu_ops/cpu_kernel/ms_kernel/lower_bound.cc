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
#include "lower_bound.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *kLowerBound = "LowerBound";

#define LOWERBOUND_COMPUTE_CASE(DTYPE, TYPE1, TYPE2, CTX)    \
  case (DTYPE): {                                            \
    uint32_t result = LowerBoundCompute<TYPE1, TYPE2>(CTX);  \
    if (result != KERNEL_STATUS_OK) {                        \
      KERNEL_LOG_ERROR("LowerBound kernel compute failed."); \
      return result;                                         \
    }                                                        \
    break;                                                   \
  }

#define LOWERBOUND_COMPUTE_CASE_ALL(TYPE, CTX)                \
  LOWERBOUND_COMPUTE_CASE(DT_INT8, int8_t, TYPE, CTX)         \
  LOWERBOUND_COMPUTE_CASE(DT_INT16, int16_t, TYPE, CTX)       \
  LOWERBOUND_COMPUTE_CASE(DT_INT32, int32_t, TYPE, CTX)       \
  LOWERBOUND_COMPUTE_CASE(DT_INT64, int64_t, TYPE, CTX)       \
  LOWERBOUND_COMPUTE_CASE(DT_UINT8, uint8_t, TYPE, CTX)       \
  LOWERBOUND_COMPUTE_CASE(DT_UINT16, uint16_t, TYPE, CTX)     \
  LOWERBOUND_COMPUTE_CASE(DT_FLOAT16, Eigen::half, TYPE, CTX) \
  LOWERBOUND_COMPUTE_CASE(DT_FLOAT, float, TYPE, CTX)         \
  LOWERBOUND_COMPUTE_CASE(DT_DOUBLE, double, TYPE, CTX)
}  // namespace

namespace aicpu {
uint32_t LowerBoundCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "LowerBound check input and output number failed.");
  Tensor *sorted_x_data = ctx.Input(0);
  Tensor *values_data = ctx.Input(1);
  Tensor *output_data = ctx.Output(0);
  auto output_type = output_data->GetDataType();
  auto sorted_x_type = sorted_x_data->GetDataType();
  auto values_type = values_data->GetDataType();
  if (sorted_x_type != values_type) {
    KERNEL_LOG_ERROR("Input[0] data type[%s] must be same with Input[1] data type[%s]", DTypeStr(sorted_x_type).c_str(),
                     DTypeStr(values_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  switch (output_type) {
    case DT_INT32:
      switch (sorted_x_type) {
        LOWERBOUND_COMPUTE_CASE_ALL(int32_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input data type[%s] not supported.", DTypeStr(sorted_x_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    case DT_INT64:
      switch (sorted_x_type) {
        LOWERBOUND_COMPUTE_CASE_ALL(int64_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input data type[%s] not supported.", DTypeStr(sorted_x_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    default:
      KERNEL_LOG_ERROR("Output data type[%s] not supported.", DTypeStr(output_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t LowerBoundCpuKernel::LowerBoundCompute(CpuKernelContext &ctx) {
  Tensor *sorted_x_data = ctx.Input(0);
  auto sorted_x_data_addr = reinterpret_cast<T1 *>(sorted_x_data->GetData());
  auto sorted_x_data_shape = sorted_x_data->GetTensorShape();
  std::vector<int64_t> sorted_x_data_shape_dims = sorted_x_data_shape->GetDimSizes();
  Tensor *values_data = ctx.Input(1);
  auto values_data_addr = reinterpret_cast<T1 *>(values_data->GetData());
  auto values_data_shape = values_data->GetTensorShape();
  int64_t values_data_num = values_data_shape->NumElements();
  std::vector<int64_t> values_data_shape_dims = values_data_shape->GetDimSizes();
  Tensor *output_data = ctx.Output(0);
  auto output_data_addr = reinterpret_cast<T2 *>(output_data->GetData());
  if (sorted_x_data_shape_dims[0] != values_data_shape_dims[0]) {
    KERNEL_LOG_ERROR("The number of rows of Input[0]:([%d]) should be consistent with that of Input[1]:([%d]).",
                     sorted_x_data_shape_dims[0], values_data_shape_dims[0]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t sorted_x_data_column = sorted_x_data_shape_dims[1];
  int64_t values_data_column = values_data_shape_dims[1];
  if (values_data_num < 1024) {
    for (int64_t i = 0; i < values_data_num; i++) {
      int64_t seq_row = i / values_data_column;
      int64_t low = seq_row * sorted_x_data_column;
      int64_t up = (seq_row + 1) * sorted_x_data_column - 1;
      int64_t mid;
      while (low <= up) {
        mid = (low + up) / 2;
        if (values_data_addr[i] <= sorted_x_data_addr[mid]) {
          up = mid - 1;
        } else {
          low = mid + 1;
        }
      }
      output_data_addr[i] = low - seq_row * sorted_x_data_column;
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t sum_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (sum_core_num > values_data_num) {
      sum_core_num = values_data_num;
    }
    auto shard_compute = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        int64_t seq_row = i / values_data_column;
        int64_t low = seq_row * sorted_x_data_column;
        int64_t up = (seq_row + 1) * sorted_x_data_column - 1;
        int64_t mid;
        while (low <= up) {
          mid = (low + up) / 2;
          if (values_data_addr[i] <= sorted_x_data_addr[mid]) {
            up = mid - 1;
          } else {
            low = mid + 1;
          }
        }
        output_data_addr[i] = low - seq_row * sorted_x_data_column;
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, values_data_num, values_data_num / sum_core_num, shard_compute),
      "LowerBound Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kLowerBound, LowerBoundCpuKernel);
}  // namespace aicpu
