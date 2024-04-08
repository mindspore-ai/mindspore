/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "cpu_kernel/ms_kernel/lgamma.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kLgammaInputNum{1};
const std::uint32_t kLgammaOutputNum{1};
const char *kLgamma{"Lgamma"};
const std::int64_t kLgammaParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename Tin, typename Tout>
inline Tout ScalarLgamma(Tin x) {
  return static_cast<Tout>(std::lgamma(x));
}

template <>
inline Eigen::half ScalarLgamma(Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(std::lgamma(static_cast<std::float_t>(x)))};
  return val;
}

inline std::uint32_t ParallelForLgamma(CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                       const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kLgammaParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename Tin, typename Tout>
inline std::uint32_t ComputeLgammaKernel(CpuKernelContext &ctx) {
  Tin *input0{static_cast<Tin *>(ctx.Input(0)->GetData())};
  Tout *output{static_cast<Tout *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForLgamma(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input0 + begin, input0 + end, output + begin, ScalarLgamma<Tin, Tout>);
  });
}

template <typename Tin, typename Tout>
inline std::uint32_t ComputeLgamma(CpuKernelContext &ctx) {
  std::uint32_t result{ComputeLgammaKernel<Tin, Tout>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "Lgamma compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckLgamma(CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data size of the input [%llu] need be the same as the output "
                          "[%llu].",
                          ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckLgamma(CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(ctx, kLgammaInputNum, kLgammaOutputNum) ? KERNEL_STATUS_PARAM_INVALID : ExtraCheckLgamma(ctx);
}

inline std::uint32_t ComputeLgamma(CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_DOUBLE:
      return ComputeLgamma<double, double>(ctx);
    case DT_FLOAT16:
      return ComputeLgamma<Eigen::half, Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeLgamma<std::float_t, std::float_t>(ctx);
    case DT_INT32:
      return ComputeLgamma<std::int32_t, std::float_t>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t LgammaCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckLgamma(ctx, kLgammaInputNum, kLgammaOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                     : detail::ComputeLgamma(ctx);
}

REGISTER_MS_CPU_KERNEL(kLgamma, LgammaCpuKernel);
}  // namespace aicpu
