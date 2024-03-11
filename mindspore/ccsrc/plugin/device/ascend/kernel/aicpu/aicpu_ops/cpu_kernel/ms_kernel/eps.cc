/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "eps.h"
#include <cstring>
#include <limits>
#include <cmath>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "Eigen/Dense"
#include "securec.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kEps = "Eps";
const int64_t kParallelDataNumCriticalPoint1 = 128 * 1024;
const int64_t kParallelDataNumCriticalPoint2 = 2 * 1024 * 1024;

#define EPS_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                               \
    uint32_t result = EpsPartCompute<TYPE>(CTX);                \
    if (result != KERNEL_STATUS_OK) {                           \
      CUST_KERNEL_LOG_ERROR(ctx, "Eps kernel compute failed."); \
      return result;                                            \
    }                                                           \
    break;                                                      \
  }
}  // namespace

namespace aicpu {
uint32_t EpsCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kEps);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    EPS_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    EPS_COMPUTE_CASE(DT_FLOAT, float, ctx)
    EPS_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx,
                            "For Eps, the supported data types are ['float16', 'float32', 'float64'], but got: [%s].",
                            DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
T getEpsilon() {
  T epsilon = static_cast<T>(0.5);
  T one = static_cast<T>(1.0);
  T two = static_cast<T>(2.0);
  while (one + epsilon / two > one) {
    epsilon = epsilon / two;
  }
  return epsilon;
}

template <typename T>
uint32_t EpsCpuKernel::EpsPartCompute(CpuKernelContext &ctx) {
  size_t data_num = static_cast<size_t>(ctx.Input(0)->NumElements());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  T min_val = getEpsilon<T>();
  if (data_num >= kParallelDataNumCriticalPoint1) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));

    if (data_num <= kParallelDataNumCriticalPoint2) {
      max_core_num = std::min(max_core_num, 4U);
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto shared_eps = [&](int64_t start, int64_t end) { SpecialEpsOutput<T>(start, end, output_data, min_val); };

    CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_eps);
  } else {
    SpecialEpsOutput<T>(0, data_num, output_data, min_val);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void EpsCpuKernel::SpecialEpsOutput(int64_t start, int64_t end, T *output_data, T value) {
  for (int64_t i = start; i < end; i++) {
    *(output_data + i) = value;
  }
}

REGISTER_MS_CPU_KERNEL(kEps, EpsCpuKernel);
}  // namespace aicpu
