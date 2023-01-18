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
#include "tanh.h"

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cmath"
#include <complex>

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kTanh = "Tanh";
constexpr int64_t kParallelDataNums = 128 * 1024;

#define Tanh_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = TanhCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Tanh kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t TanhCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kTanh);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    Tanh_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
      Tanh_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx) Tanh_COMPUTE_CASE(DT_FLOAT, float, ctx)
        Tanh_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx) Tanh_COMPUTE_CASE(DT_DOUBLE, double, ctx) default
        : KERNEL_LOG_ERROR("Tanh kernel data type [%s] not support.", DTypeStr(data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TanhCpuKernel::TanhCompute(CpuKernelContext &ctx) {
  Eigen::internal::scalar_tanh_op<T> tanh_op;
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  size_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if (data_size <= kParallelDataNums) {
    for (size_t i = 0; i < data_num; i++) {
      auto x_idx = input_x + i;  // i-th value of input0
      *(output_y + i) = tanh_op((*x_idx));
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    auto shard_Tanh = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto x_idx = input_x + i;  // i-th value of input0
        *(output_y + i) = tanh_op((*x_idx));
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_Tanh),
                        "Tanh Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTanh, TanhCpuKernel);
}  // namespace aicpu
