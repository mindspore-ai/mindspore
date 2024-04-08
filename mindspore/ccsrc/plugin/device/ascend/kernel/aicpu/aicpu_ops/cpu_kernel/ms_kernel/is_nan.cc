/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "cpu_kernel/ms_kernel/is_nan.h"

#include <Eigen/Dense>
#include <algorithm>

#include "utils/kernel_util.h"
#include "inc/kernel_log.h"
#include "cpu_types.h"
#include "context/common/status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/inc/cpu_kernel_utils.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const KIsNan = "IsNan";
constexpr int64_t kParallelDataNumsFloat16 = 32 * 1024;
constexpr int64_t kParallelDataNumsFloat = 256 * 1024;
constexpr int64_t kParallelDataNumsDouble = 256 * 1024;

#define ISNAN_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                 \
    uint32_t result = IsNanCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                             \
      CUST_KERNEL_LOG_ERROR(ctx, "IsNan kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }
}  // namespace

namespace aicpu {
uint32_t IsNanCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", KIsNan)
  CUST_KERNEL_HANDLE_ERROR(ctx, IsNanCheck(ctx), "[%s] check params failed.", KIsNan);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ISNAN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    ISNAN_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ISNAN_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "IsNan kernel data type [%s] not supports.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t IsNanCpuKernel::IsNanCheck(CpuKernelContext &ctx) const {
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IsNanCpuKernel::IsNanCompute(CpuKernelContext &ctx) const {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<bool *>(ctx.Output(0)->GetData());

  auto data_type = ctx.Input(0)->GetDataType();
  int64_t data_num = ctx.Output(0)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));

  if ((data_type == DT_FLOAT16 && data_size <= kParallelDataNumsFloat16) ||
      (data_type == DT_FLOAT && data_size <= kParallelDataNumsFloat) ||
      (data_type == DT_DOUBLE && data_size <= kParallelDataNumsDouble)) {
    for (int64_t index = 0; index < data_num; index++) {
      *(output + index) = Eigen::numext::isnan(*(input + index));
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto shard_isinf = [&](size_t start, size_t end) {
      for (size_t index = start; index < end; index++) {
        *(output + index) = Eigen::numext::isnan(*(input + index));
      }
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_isinf),
                             "IsInf Compute failed.");
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(KIsNan, IsNanCpuKernel);
}  // namespace aicpu
