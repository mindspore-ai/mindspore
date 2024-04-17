/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "cpu_kernel/ms_kernel/sign.h"
#include <type_traits>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kSign = "Sign";
constexpr int64_t kParallelDataNums = 128 * 1024;

#define SIGN_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                \
    uint32_t result = SignCompute<TYPE>(CTX);                    \
    if (result != static_cast<uint32_t>(KERNEL_STATUS_OK)) {     \
      CUST_KERNEL_LOG_ERROR(ctx, "Sign kernel compute failed."); \
      return result;                                             \
    }                                                            \
    break;                                                       \
  }

#define SIGN_COMPUTE_CASE2(DTYPE, TYPE, CTX)                     \
  case (DTYPE): {                                                \
    uint32_t result = SignComputeComplex<TYPE>(CTX);             \
    if (result != static_cast<uint32_t>(KERNEL_STATUS_OK)) {     \
      CUST_KERNEL_LOG_ERROR(ctx, "Sign kernel compute failed."); \
      return result;                                             \
    }                                                            \
    break;                                                       \
  }
}  // namespace

namespace aicpu {
uint32_t SignCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kSign);
  CUST_KERNEL_HANDLE_ERROR(ctx, static_cast<uint32_t>(SignCheck(ctx)), "[%s] check params failed.", kSign);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SIGN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SIGN_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SIGN_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    SIGN_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SIGN_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SIGN_COMPUTE_CASE2(DT_COMPLEX64, std::complex<float>, ctx)
    SIGN_COMPUTE_CASE2(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Sign kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

KernelStatus SignCpuKernel::SignCheck(CpuKernelContext &ctx) const {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input tensor shape failed.")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SignCpuKernel::SignCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
  if (data_size <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      if (*(input_x + i) > static_cast<T>(0)) {
        *(output_y + i) = static_cast<T>(1);
      } else if (*(input_x + i) < static_cast<T>(0)) {
        *(output_y + i) = static_cast<T>(-1);
      } else {
        *(output_y + i) = static_cast<T>(0);
      }
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_sign = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (*(input_x + i) > static_cast<T>(0)) {
          *(output_y + i) = static_cast<T>(1);
        } else if (*(input_x + i) < static_cast<T>(0)) {
          *(output_y + i) = static_cast<T>(-1);
        } else {
          *(output_y + i) = static_cast<T>(0);
        }
      }
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_sign),
                             "Sign Compute failed.");
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T>
uint32_t SignCpuKernel::SignComputeComplex(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
  if (data_size <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      if (*(input_x + i) != static_cast<T>(0)) {
        *(output_y + i) = (*(input_x + i) / Eigen::numext::abs(*(input_x + i)));
      } else {
        *(output_y + i) = static_cast<T>(0);
      }
    }
  } else {
    uint32_t min_num = 1;
    int64_t max_core_num = std::max(min_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_sign = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (*(input_x + i) != static_cast<T>(0)) {
          *(output_y + i) = (*(input_x + i) / Eigen::numext::abs(*(input_x + i)));
        } else {
          *(output_y + i) = static_cast<T>(0);
        }
      }
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_sign),
                             "Sign Compute failed.");
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}
REGISTER_CPU_KERNEL(kSign, SignCpuKernel);
}  // namespace aicpu
