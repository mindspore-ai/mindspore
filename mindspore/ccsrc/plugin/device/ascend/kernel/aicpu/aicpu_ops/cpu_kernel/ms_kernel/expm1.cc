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
#include "expm1.h"

#include "cpu_kernel_utils.h"
#include "math.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#include <math.h>
#include <iostream>

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kExpm1 = "Expm1";

#define EXPM1_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                       \
    uint32_t result = Expm1Compute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Expm1 kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }

#define EXPM1_COMPUTE_CASE2(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                       \
    uint32_t result = Expm1Compute2<TYPE>(CTX);         \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Expm1 kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }

#define EXPM1_COMPUTE_CASE3(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                       \
    uint32_t result = Expm1Compute3<TYPE>(CTX);         \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Expm1 kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }
}  // namespace

namespace aicpu {
uint32_t Expm1CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kExpm1);
  KERNEL_HANDLE_ERROR(Expm1Check(ctx), "[%s] check params failed.", kExpm1);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    EXPM1_COMPUTE_CASE2(DT_FLOAT16, Eigen::half, ctx)
    EXPM1_COMPUTE_CASE3(DT_FLOAT, float, ctx)
    EXPM1_COMPUTE_CASE3(DT_DOUBLE, double, ctx)
    EXPM1_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    EXPM1_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Expm1 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t Expm1CpuKernel::Expm1Check(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t Expm1CpuKernel::Expm1Compute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto data_type = ctx.Input(0)->GetDataType();
  size_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  T num0 = static_cast<T>(-1.0);

  if (((data_type = DT_COMPLEX64) && (data_size <= 64 * 1024)) ||
      ((data_type = DT_COMPLEX128) && (data_size <= 64 * 1024))) {
    for (size_t i = 0; i < data_num; i++) {
      (*(output_y + i)) = Eigen::numext::exp(*(input_x + i)) + num0;
    }
  } else {
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_expm1 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        (*(output_y + i)) = Eigen::numext::exp(*(input_x + i)) + num0;
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_expm1),
                        "Expm1 Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t Expm1CpuKernel::Expm1Compute2(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<Eigen::half *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<Eigen::half *>(ctx.Output(0)->GetData());
  size_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(Eigen::half);
  Eigen::half num0 = static_cast<Eigen::half>(-1.0);
  if (data_size <= 32 * 1024) {
    for (size_t i = 0; i < data_num; i++) {
      *(output_y + i) = exp(*(input_x + i)) + num0;
    }
  } else {
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_expm1 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        *(output_y + i) = exp(*(input_x + i)) + num0;
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_expm1),
                        "Expm1 Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t Expm1CpuKernel::Expm1Compute3(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto data_type = ctx.Input(0)->GetDataType();
  size_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if ((data_type == DT_DOUBLE && data_size <= 64 * 1024) || (data_type == DT_FLOAT && data_size <= 16 * 1024)) {
    for (size_t i = 0; i < data_num; i++) {
      *(output_y + i) = expm1(*(input_x + i));
    }
  } else {
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_expm1 = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        *(output_y + i) = expm1(*(input_x + i));
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_expm1),
                        "Expm1 Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kExpm1, Expm1CpuKernel);
}  // namespace aicpu