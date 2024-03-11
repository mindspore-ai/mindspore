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
#include "cpu_kernel/ms_kernel/log1p.h"

#include <algorithm>
#include <limits>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kLog1p = "Log1p";
constexpr int64_t kParallelDataNums = 16 * 1024;

#define LOG1P_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                 \
    uint32_t result = Log1pCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                             \
      CUST_KERNEL_LOG_ERROR(ctx, "Log1p kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }

#define LOG1P_COMPUTE_CASE2(DTYPE, TYPE, CTX)                     \
  case (DTYPE): {                                                 \
    uint32_t result = Log1pComputeComplex<TYPE>(CTX);             \
    if (result != KERNEL_STATUS_OK) {                             \
      CUST_KERNEL_LOG_ERROR(ctx, "Log1p kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }
}  // namespace

namespace aicpu {
uint32_t Log1pCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kLog1p);
  CUST_KERNEL_HANDLE_ERROR(ctx, Log1pCheck(ctx), "[%s] check params failed.", kLog1p);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    LOG1P_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    LOG1P_COMPUTE_CASE(DT_FLOAT, float, ctx)
    LOG1P_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    LOG1P_COMPUTE_CASE2(DT_COMPLEX64, std::complex<float>, ctx)
    LOG1P_COMPUTE_CASE2(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Log1p kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t Log1pCpuKernel::Log1pCheck(CpuKernelContext &ctx) const {
  auto input_0 = ctx.Input(0);
  auto output_0 = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input tensor shape failed.")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t Log1pCpuKernel::Log1pCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
  if (data_size <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      auto val = *(input_x + i);
      if (val < static_cast<T>(-1)) {
        *(output_y + i) = std::numeric_limits<T>::quiet_NaN();
      } else {
        *(output_y + i) = Eigen::numext::log1p(val);
      }
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_log1p = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto val = *(input_x + i);
        if (val < static_cast<T>(-1)) {
          *(output_y + i) = std::numeric_limits<T>::quiet_NaN();
        } else {
          *(output_y + i) = Eigen::numext::log1p(val);
        }
      }
      return KERNEL_STATUS_OK;
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_log1p),
                             "Log1p Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t Log1pCpuKernel::Log1pComputeComplex(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto data_type = ctx.Input(0)->GetDataType();
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
  typedef Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> ArrayxXd;
  ArrayxXd array_x(1, data_num);
  if (data_size <= kParallelDataNums) {
    if (data_type == DT_COMPLEX64) {
      for (int64_t i = 0; i < data_num; i++) {
        array_x(0, i) = *(input_x + i);
        CUST_KERNEL_CHECK_FALSE(ctx, array_x(0, i).real() >= static_cast<float>(-1), KERNEL_STATUS_PARAM_INVALID,
                                "[%llu] must be at least more than -1.", i);
        *(output_y + i) = Eigen::numext::log1p(*(input_x + i));
      }
    } else {
      for (int64_t i = 0; i < data_num; i++) {
        array_x(0, i) = *(input_x + i);
        CUST_KERNEL_CHECK_FALSE(ctx, array_x(0, i).real() >= static_cast<double>(-1), KERNEL_STATUS_PARAM_INVALID,
                                "[%llu] must be at least more than -1.", i);
        *(output_y + i) = Eigen::numext::log1p(*(input_x + i));
      }
    }
    return KERNEL_STATUS_OK;
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_log1p = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if (data_type == DT_COMPLEX64) {
          array_x(0, i) = *(input_x + i);
          CUST_KERNEL_CHECK_FALSE(ctx, array_x(0, i).real() >= static_cast<float>(-1), KERNEL_STATUS_PARAM_INVALID,
                                  "[%llu] must be at least more than -1.", i);
          *(output_y + i) = Eigen::numext::log1p(*(input_x + i));
        } else {
          array_x(0, i) = *(input_x + i);
          CUST_KERNEL_CHECK_FALSE(ctx, array_x(0, i).real() >= static_cast<double>(-1), KERNEL_STATUS_PARAM_INVALID,
                                  "[%llu] must be at least more than -1.", i);
          *(output_y + i) = Eigen::numext::log1p(*(input_x + i));
        }
      }
      return KERNEL_STATUS_OK;
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_log1p),
                             "Log1p Compute failed.");
    return KERNEL_STATUS_OK;
  }
}
REGISTER_MS_CPU_KERNEL(kLog1p, Log1pCpuKernel);
}  // namespace aicpu
