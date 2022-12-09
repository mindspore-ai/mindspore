/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#include "cpu_kernel/ms_kernel/acos.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kAcosInputNum{1u};
const std::uint32_t kAcosOutputNum{1u};
const char *const kAcos{"Acos"};
const std::int64_t kAcosParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarAcos(const T x) {
  return std::acos(x);
}

template <>
inline Eigen::half ScalarAcos(const Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(std::acos(static_cast<std::float_t>(x)))};
  return val;
}

inline std::uint32_t ParallelForAcos(const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                     const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kAcosParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAcosKernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForAcos(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input0 + begin, input0 + end, output + begin, ScalarAcos<T>);
  });
}

template <typename T>
inline std::uint32_t ComputeAcos(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAcosKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Acos compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAcos(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
      "The data size of the input [%llu] need be the same as the output "
      "[%llu].",
      ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAcos(const CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num) ? KERNEL_STATUS_PARAM_INVALID : ExtraCheckAcos(ctx);
}

inline std::uint32_t ComputeAcos(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAcos<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAcos<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAcos<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AcosCpuKernel::Compute(const CpuKernelContext &ctx) {
  return detail::CheckAcos(ctx, kAcosInputNum, kAcosOutputNum) ? KERNEL_STATUS_PARAM_INVALID : detail::ComputeAcos(ctx);
}

REGISTER_CPU_KERNEL(kAcos, AcosCpuKernel);
}  // namespace aicpu
