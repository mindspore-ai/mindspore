/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "inc/cpu_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "topkrouter.h"
#include <vector>

namespace {
const uint32_t kOutputNum = 2;
const uint32_t kInputNum = 3;
const char *const kTopKRouter = "TopKRouter";
#define TOPKROUTER_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                      \
    uint32_t result = TopKRouterCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "TopKRouter kernel compute failed."); \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }

}  // namespace

namespace aicpu {
uint32_t TopKRouterCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "TopKRouter check input and output number failed.");
  auto output_type = ctx.Output(0)->GetDataType();
  switch (output_type) {
    TOPKROUTER_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TOPKROUTER_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Output data type [%s] not support.", DTypeStr(output_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TopKRouterCpuKernel::TopKRouterCompute(const CpuKernelContext &ctx) {
  auto input_data = static_cast<T *>(ctx.Input(0)->GetData());
  auto capacity_ptr = static_cast<int64_t *>(ctx.Input(1)->GetData());
  auto expert_num_ptr = static_cast<int64_t *>(ctx.Input(2)->GetData());
  auto dispatch_index = static_cast<T *>(ctx.Output(0)->GetData());
  auto combine_index = static_cast<T *>(ctx.Output(1)->GetData());
  auto input_shape = ctx.Input(0)->GetTensorShape();
  auto batch = input_shape->GetDimSize(0);
  auto length = input_shape->GetDimSize(1);
  auto k = input_shape->GetDimSize(2);
  auto capacity = *capacity_ptr;
  auto expert_num = *expert_num_ptr;

  // init dispatch index
  auto dispatch_shape = ctx.Output(0)->GetTensorShape();
  auto dispatch_num = dispatch_shape->NumElements();
  for (int i = 0; i < dispatch_num; i++) {
    dispatch_index[i] = 0;
  }
  // init counter
  std::vector<int64_t> expert_counter(batch * expert_num, 0);

  for (int bs = 0; bs < batch; bs++) {
    for (int i = 0; i < length; i++) {
      for (int j = 0; j < k; j++) {
        auto token_index = i;
        auto expert_id = input_data[bs * length * k + i * k + j];
        auto position_in_expert = expert_counter[bs * expert_num + expert_id];
        if (position_in_expert < capacity) {
          dispatch_index[bs * expert_num * capacity + expert_id * capacity + position_in_expert] =
            static_cast<T>(token_index + 1);
          combine_index[bs * length * k + i * k + j] =
            static_cast<T>(expert_id * (capacity + 1) + position_in_expert + 1);
          expert_counter[bs * expert_num + expert_id] = static_cast<T>(position_in_expert + 1);
        } else {
          combine_index[bs * length * k + i * k + j] = static_cast<T>(expert_id * (capacity + 1));
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kTopKRouter, TopKRouterCpuKernel);

}  // namespace aicpu
