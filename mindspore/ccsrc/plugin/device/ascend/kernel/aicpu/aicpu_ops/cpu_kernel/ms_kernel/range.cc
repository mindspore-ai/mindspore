/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "range.h"
#include <cmath>
#include <thread>
#include <numeric>
#include <vector>
#include <functional>
#include "utils/atomic_op.h"
#include "utils/kernel_util.h"
#include "inc/kernel_log.h"

namespace aicpu {
namespace {
constexpr auto kRangeInputNum = 3;
constexpr auto kRangeOutputNum = 1;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;
constexpr auto kDim3 = 3;
const char *kRange = "Range";

template <typename T>
T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}
}  // namespace

template <typename T>
uint32_t RangeKernel::RangeTask(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kRangeInputNum, kRangeOutputNum), "NormalCheck failed.");
  auto start = reinterpret_cast<T *>(ctx.Input(0))[0];
  auto limit = reinterpret_cast<T *>(ctx.Input(1))[0];
  auto delta = reinterpret_cast<T *>(ctx.Input(2))[0];
  auto output = reinterpret_cast<T *>(ctx.Output(0));
  if (Sign(delta) * Sign(limit - start) >= 0) {
    size_t output_size = 0;
    if (std::is_integral<T>::value) {
      output_size = static_cast<size_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
    } else {
      output_size = static_cast<size_t>(std::ceil((limit - start) / delta));
    }
    for (int index = 0; index < SizeToInt(ctx, output_size); index++) {
      output[index] = delta * index + start;
    }
  } else {
    CUST_KERNEL_LOG_ERROR(ctx, "Invalid delta size.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t RangeKernel::Compute(CpuKernelContext &ctx) {
  auto input_tensor = ctx.Input(0);
  index_type_ = input_tensor->GetDataType();
  switch (index_type_) {
    case DT_INT32:
      return RangeTask<int>(ctx);
    case DT_INT64:
      return RangeTask<int64_t>(ctx);
    case DT_FLOAT:
      return RangeTask<float>(ctx);
    case DT_DOUBLE:
      return RangeTask<double>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Input data type not supported: %s", DTypeStr(index_type_).c_str());
      return KERNEL_STATUS_INNER_ERROR;
  }
}
REGISTER_MS_CPU_KERNEL(kRange, RangeKernel);
}  // namespace aicpu