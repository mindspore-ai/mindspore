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

#include "ms_kernel/relu.h"
#include <algorithm>
#include "common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kRelu = "Relu";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define RELU_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = ReluCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Relu kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
}  // namespace

namespace aicpu {
uint32_t ReluCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Relu check input and output number failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    RELU_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    RELU_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    RELU_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    RELU_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    RELU_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    RELU_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    RELU_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    RELU_COMPUTE_CASE(DT_FLOAT, float, ctx)
    RELU_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Relu kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
void ReluCpuKernel::DoCompute(int64_t start, int64_t end, const T *input1, T *output) {
  for (int64_t i = start; i < end; ++i) {
    T v = *(input1 + i);
    bool p = v > static_cast<T>(0);
    *(output + i) = p ? v : static_cast<T>(0);
  }
}

template <typename T>
uint32_t ReluCpuKernel::ReluCompute(const CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_relu = [&](int64_t start, int64_t end) { DoCompute<T>(start, end, in0, out); };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_relu),
                        "Relu Compute failed.");
  } else {
    DoCompute<T>(0, data_num, in0, out);
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kRelu, ReluCpuKernel);
}  // namespace aicpu
