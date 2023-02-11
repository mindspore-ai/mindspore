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
#include "imag.h"
#include "Eigen/Eigen"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kImag = "Imag";
constexpr int64_t kFolatDataNums = 8 * 128 * 1024;
constexpr int64_t kDoubleDataNums = 16 * 128 * 1024;

#define Imag_COMPUTE_CASE(IN_DTYPE, IN_TYPE, OUT_DTYPE, CTX)                                             \
  case (IN_DTYPE): {                                                                                     \
    switch (OUT_DTYPE) {                                                                                 \
      case (DT_FLOAT): {                                                                                 \
        uint32_t result = ImagCompute<IN_TYPE, float>(CTX);                                              \
        if (result != KERNEL_STATUS_OK) {                                                                \
          KERNEL_LOG_ERROR("Imag kernel compute failed.");                                               \
          return result;                                                                                 \
        }                                                                                                \
        break;                                                                                           \
      }                                                                                                  \
      case (DT_DOUBLE): {                                                                                \
        uint32_t result = ImagCompute<IN_TYPE, double>(CTX);                                             \
        if (result != KERNEL_STATUS_OK) {                                                                \
          KERNEL_LOG_ERROR("Imag kernel compute failed.");                                               \
          return result;                                                                                 \
        }                                                                                                \
        break;                                                                                           \
      }                                                                                                  \
      default:                                                                                           \
        KERNEL_LOG_ERROR("Imag kernel output data type [%s] not support.", DTypeStr(OUT_DTYPE).c_str()); \
        return KERNEL_STATUS_PARAM_INVALID;                                                              \
    }                                                                                                    \
    break;                                                                                               \
  }
}  // namespace

namespace aicpu {
uint32_t ImagCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kImag);
  DataType input_type = ctx.Input(0)->GetDataType();
  switch (input_type) {
    Imag_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, DT_FLOAT, ctx)
      Imag_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, DT_DOUBLE, ctx) default
        : KERNEL_LOG_ERROR("Imag kernel input data type [%s] not support.", DTypeStr(input_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename t>
uint32_t ImagCpuKernel::ImagCompute(CpuKernelContext &ctx) {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<t *>(ctx.Output(0)->GetData());

  auto data_type = ctx.Input(0)->GetDataType();
  int64_t data_num = ctx.Output(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if ((data_type == DT_COMPLEX64 && data_size <= kFolatDataNums) ||
      (data_type == DT_COMPLEX128 && data_size <= kDoubleDataNums)) {
    for (int64_t index = 0; index < data_num; ++index) {
      *(output + index) = (*(input + index)).imag();
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shard_imag = [&](size_t start, size_t end) {
      for (size_t index = start; index < end; ++index) {
        *(output + index) = (*(input + index)).imag();
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_imag),
                        "imag Compute failed");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kImag, ImagCpuKernel);
}  // namespace aicpu
