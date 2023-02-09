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
#include "mvlgamma.h"

#include "cpu_kernel_utils.h"
#include "igamma_utils.cc"
#include "igamma_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kMvlgamma = "Mvlgamma";

#define MVLGAMMA_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                          \
    uint32_t result = MvlgammaCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("Mvlgamma kernel compute failed."); \
      return result;                                       \
    }                                                      \
    break;                                                 \
  }

constexpr double HALF = 0.5;
constexpr double QUARTER = 0.25;
}  // namespace

namespace aicpu {
uint32_t MvlgammaCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(MvlgammaCheck(ctx), "Mvlgamma check params failed.");

  const Tensor *input_x = ctx.Input(0);
  auto data_type = input_x->GetDataType();

  switch (data_type) {
    MVLGAMMA_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MVLGAMMA_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Mvlgamma kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t MvlgammaCpuKernel::MvlgammaCheck(CpuKernelContext &ctx) {
  // check input, output and attr not null
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("p"), KERNEL_STATUS_PARAM_INVALID, "Get attr failed.")
  NormalCheck(ctx, 1, 1, {"p"});

  // check input and output datatype as the same
  DataType input_datatype = ctx.Input(0)->GetDataType();
  DataType output_datatype = ctx.Output(0)->GetDataType();
  KERNEL_CHECK_FALSE((input_datatype == output_datatype), KERNEL_STATUS_PARAM_INVALID,
                     "Input data type[%d] must be the same as Output data type[%d].", input_datatype, output_datatype)

  auto attr_value = ctx.GetAttr("p")->GetInt();
  KERNEL_CHECK_FALSE((attr_value >= 1), KERNEL_STATUS_PARAM_INVALID, "p has to be greater than or equal to 1[%lld]",
                     attr_value)  // 已经用GetAttr获取

  KERNEL_LOG_INFO("MvlgammaCpuKernel[%s], input: size[%llu], output: size[%llu].", ctx.GetOpType().c_str(),
                  ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
T MvlgammaCpuKernel::MvlgammaSingle(T &x, const int &p, bool &error) {
  if (!(x > HALF * (p - 1))) {
    error = true;
    KERNEL_LOG_ERROR("All elements of `x` must be greater than (p-1)/2");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const auto p2_sub_p = static_cast<T>(p * (p - 1));
  T output = p2_sub_p * std::log(M_PI) * QUARTER;
  for (int i = 0; i < p; i++) {
    output += Lgamma(x - HALF * i);
  }
  return output;
}

template <typename T>
uint32_t MvlgammaCpuKernel::MvlgammaCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto attr_p = ctx.GetAttr("p")->GetInt();

  auto input0_shape = ctx.Input(0)->GetTensorShape();
  int64_t data_num = input0_shape->NumElements();
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }

  bool error = false;
  auto shard_mvlgamma = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      *(output_y + i) = MvlgammaSingle<T>(*(input_x + i), attr_p, error);
    }
  };

  if (max_core_num == 0) {
    KERNEL_LOG_ERROR("max_core_num could not be 0,");
  }
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_mvlgamma),
                      "Mvlgamma Compute failed.");
  if (error == true) {
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    return KERNEL_STATUS_OK;
  }
}

REGISTER_CPU_KERNEL(kMvlgamma, MvlgammaCpuKernel);
}  // namespace aicpu
