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
#include "cpu_kernel/ms_kernel/complex.h"

#include <algorithm>

#include "Eigen/Eigen"

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kComplex = "Complex";
constexpr int64_t kFloatMaxNums = 8 * 128 * 1024;
constexpr int64_t kDoubleMaxNums = 16 * 128 * 1024;
#define Complex_COMPUTE_CASE(IN_DTYPE, IN_TYPE, OUT_DTYPE, CTX)                                                       \
  case (IN_DTYPE): {                                                                                                  \
    switch (OUT_DTYPE) {                                                                                              \
      case (DT_COMPLEX64): {                                                                                          \
        uint32_t result = ComplexCompute<float, std::complex<float>>(CTX);                                            \
        if (result != KERNEL_STATUS_OK) {                                                                             \
          CUST_KERNEL_LOG_ERROR(ctx, "Complex kernel compute failed.");                                               \
          return result;                                                                                              \
        }                                                                                                             \
        break;                                                                                                        \
      }                                                                                                               \
      case (DT_COMPLEX128): {                                                                                         \
        uint32_t result = ComplexCompute<double, std::complex<double>>(CTX);                                          \
        if (result != KERNEL_STATUS_OK) {                                                                             \
          CUST_KERNEL_LOG_ERROR(ctx, "Complex kernel compute failed.");                                               \
          return result;                                                                                              \
        }                                                                                                             \
        break;                                                                                                        \
      }                                                                                                               \
      default:                                                                                                        \
        CUST_KERNEL_LOG_ERROR(ctx, "Complex kernel output data type [%s] not support.", DTypeStr(OUT_DTYPE).c_str()); \
        return KERNEL_STATUS_PARAM_INVALID;                                                                           \
    }                                                                                                                 \
    break;                                                                                                            \
  }
}  // namespace

namespace aicpu {
uint32_t ComplexCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(const_cast<CpuKernelContext &>(ctx), kInputNum, kOutputNum),
                           "[%s] check input and output failed.", kComplex);
  DataType input_type = ctx.Input(0)->GetDataType();
  switch (input_type) {
    Complex_COMPUTE_CASE(DT_FLOAT, float, DT_COMPLEX64, ctx)
      Complex_COMPUTE_CASE(DT_DOUBLE, double, DT_COMPLEX128, ctx) default
        : CUST_KERNEL_LOG_ERROR(ctx, "Complex kernel input data type [%s] not support.", DTypeStr(input_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
template <typename T, typename t>
uint32_t ComplexCpuKernel::ComplexCompute(CpuKernelContext &ctx) {
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<t *>(ctx.Output(0)->GetData());
  auto data_type = ctx.Input(0)->GetDataType();
  int64_t data_num = ctx.Output(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if ((data_type == DT_FLOAT && data_size <= kFloatMaxNums) ||
      (data_type == DT_DOUBLE && data_size <= kDoubleMaxNums)) {
    for (int64_t index = 0; index < data_num; ++index) {
      *(output + index) = t(*(input0 + index), *(input1 + index));
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_complex = [&](size_t start, size_t end) {
      for (size_t index = start; index < end; ++index) {
        *(output + index) = t(*(input0 + index), *(input1 + index));
      }
    };
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_complex),
                             "complex Compute failed");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kComplex, ComplexCpuKernel);
}  // namespace aicpu
