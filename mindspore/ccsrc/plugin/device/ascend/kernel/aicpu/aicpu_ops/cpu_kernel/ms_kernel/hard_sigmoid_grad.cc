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
#include "cpu_kernel/ms_kernel/hard_sigmoid_grad.h"

#include <algorithm>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *const kHardSigmoidGrad = "HardSigmoidGrad";
const int64_t kParallelDataNums = 16 * 1024;

#define HARD_SIGMOID_GRAD_COMPUTE_CASE(DTYPE1, TYPE1, TYPE2, CTX) \
  case (DTYPE1): {                                                \
    uint32_t result = HardSigmoidGradCompute<TYPE1, TYPE2>(CTX);  \
    if (result != KERNEL_STATUS_OK) {                             \
      KERNEL_LOG_ERROR("HardSigmoidGrad kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }
}  // namespace

namespace aicpu {
uint32_t HardSigmoidGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kHardSigmoidGrad);
  DataType grads_type = ctx.Input(0)->GetDataType();
  DataType x_type = ctx.Input(1)->GetDataType();
  if (grads_type != x_type) {
    KERNEL_LOG_ERROR("HardSigmoidGrad kernel input[0] data type [%s] must be the same as input[1] data type [%s].",
                     DTypeStr(grads_type).c_str(), DTypeStr(x_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  switch (grads_type) {
    HARD_SIGMOID_GRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, Eigen::half, ctx)
    HARD_SIGMOID_GRAD_COMPUTE_CASE(DT_FLOAT, float, float, ctx)
    HARD_SIGMOID_GRAD_COMPUTE_CASE(DT_DOUBLE, double, double, ctx)
    default:
      KERNEL_LOG_ERROR("HardSigmoidGrad kernel inputs data type [%s] not support.", DTypeStr(grads_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t HardSigmoidGradCpuKernel::HardSigmoidGradCompute(const CpuKernelContext &ctx) {
  auto grads = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto input_x = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());
  auto y = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(1)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T2));
  const T2 zero = static_cast<T2>(0);
  const T2 three = static_cast<T2>(3);
  const T2 neg_three = static_cast<T2>(-3);
  const T2 one_sixth = static_cast<T2>(1.0f / 6.0f);
  if (data_size <= kParallelDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      *(y + i) =
        (*(input_x + i) > neg_three && *(input_x + i) < three) ? static_cast<T2>(*(grads + i)) * one_sixth : zero;
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    int64_t perUnitSize = max_core_num > 0 ? data_num / max_core_num : data_num;
    auto shard_hard_sigmoid_grad = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        *(y + i) =
          (*(input_x + i) > neg_three && *(input_x + i) < three) ? static_cast<T2>(*(grads + i)) * one_sixth : zero;
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, perUnitSize, shard_hard_sigmoid_grad),
                        "HardSigmoidGrad Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kHardSigmoidGrad, HardSigmoidGradCpuKernel);
}  // namespace aicpu
