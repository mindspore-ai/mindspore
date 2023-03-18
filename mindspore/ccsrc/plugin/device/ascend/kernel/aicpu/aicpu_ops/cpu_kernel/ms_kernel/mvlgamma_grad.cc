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
#include "mvlgamma_grad.h"

#include "cpu_kernel_utils.h"
#include "igamma_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kMvlgammaGrad = "MvlgammaGrad";

#define MVLGAMMAGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                              \
    uint32_t result = MvlgammaGradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("MvlgammaGrad kernel compute failed."); \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }

constexpr double HALF = 0.5;
constexpr double QUARTER = 0.25;
}  // namespace

namespace aicpu {
uint32_t MvlgammaGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(MvlgammaGradCheck(ctx), "MvlgammaGrad check params failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MVLGAMMAGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MVLGAMMAGRAD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("MvlgammaGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t MvlgammaGradCpuKernel::MvlgammaGradCheck(CpuKernelContext &ctx) {
  // check input, output and attr not null
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(1)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("p"), KERNEL_STATUS_PARAM_INVALID, "Get attr failed.")
  NormalCheck(ctx, 2, 1, {"p"});

  // check input and output datatype as the same
  DataType input0_type = ctx.Input(0)->GetDataType();
  DataType input1_type = ctx.Input(1)->GetDataType();
  DataType output_type = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%d] need be same with "
                     "input1 [%d].",
                     input0_type, input1_type)
  KERNEL_CHECK_FALSE((input0_type == output_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%d] need be same with "
                     "output [%d].",
                     input0_type, output_type)

  auto attr_value = ctx.GetAttr("p")->GetInt();
  KERNEL_CHECK_FALSE((attr_value >= 1), KERNEL_STATUS_PARAM_INVALID, "p has to be greater than or equal to 1[%lld]",
                     attr_value)  // 已经用GetAttr获取

  KERNEL_LOG_INFO(
    "MvlgammaGradCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize(), ctx.Output(0)->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
T MvlgammaGradCpuKernel::MvlgammaGradSingle(T &y_grad, T &x, const int &p) {
  T output = 0;
  for (int i = 0; i < p; i++) {
    output += Digamma(x - HALF * i);
  }
  output *= y_grad;
  return output;
}

template <typename T>
uint32_t MvlgammaGradCpuKernel::MvlgammaGradCompute(CpuKernelContext &ctx) {
  auto input_y_grad = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_x = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output_x_grad = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto attr_p = ctx.GetAttr("p")->GetInt();

  auto input0_shape = ctx.Input(0)->GetTensorShape();
  int64_t data_num = input0_shape->NumElements();
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }

  auto shard_mvlgammagrad = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      *(output_x_grad + i) = MvlgammaGradSingle<T>(*(input_y_grad + i), *(input_x + i), attr_p);
    }
  };

  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num could not be 0,");
  }
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_mvlgammagrad),
                      "MvlgammaGrad Compute failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMvlgammaGrad, MvlgammaGradCpuKernel);
}  // namespace aicpu
