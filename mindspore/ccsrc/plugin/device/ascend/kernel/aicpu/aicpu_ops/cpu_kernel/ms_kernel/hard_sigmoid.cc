/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All right reserved.
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
#include "cpu_kernel/ms_kernel/hard_sigmoid.h"

#include <algorithm>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kHardSigmoid = "HardSigmoid";
const int64_t kParallelDataNums = 16 * 1024;
const float alpha = 0.16666666;
const float beta = 0.5;

#define HARD_SIGMOID_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                             \
    uint32_t result = HardSigmoidCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                         \
      KERNEL_LOG_ERROR("HardSigmoid kernel compute failed."); \
      return result;                                          \
    }                                                         \
    break;                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t HardSigmoidCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kHardSigmoid);
  DataType data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    HARD_SIGMOID_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    HARD_SIGMOID_COMPUTE_CASE(DT_FLOAT, float, ctx)
    HARD_SIGMOID_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("HardSigmoid kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t HardSigmoidCpuKernel::HardSigmoidCompute(const CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
  const T zero = static_cast<T>(0);
  const T three = static_cast<T>(3);
  const T six = static_cast<T>(6);
  if (data_size <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      *(output_y + i) = std::min(std::max(*(input_x + i) + three, zero), six) / six;
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    int64_t perUnitSize = max_core_num > 0 ? data_num / max_core_num : data_num;
    auto shard_hard_sigmoid = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        *(output_y + i) = std::min(std::max(*(input_x + i) + three, zero), six) / six;
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, perUnitSize, shard_hard_sigmoid),
                        "HardSigmoid Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kHardSigmoid, HardSigmoidCpuKernel);
}  // namespace aicpu
