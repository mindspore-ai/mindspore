/**
 * Copyright 2021 Jilin University
 * Copyright 2020 Huawei Technologies Co., Ltd.
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

#include "cpu_kernel/ms_kernel/add_v2.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "common/kernel_log.h"
#include "cpu_kernel/common/status.h"
#include "utils/bcast.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/inc/cpu_context.h"

namespace {
const std::uint32_t kAddV2InputNum{2u};
const std::uint32_t kAddV2OutputNum{1u};
const char *kAddV2{"AddV2"};
const std::int64_t kAddV2ParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarAddV2(T a, T b) {
  return a + b;
}
template <typename T>
inline std::uint32_t ParallelForAddV2(const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                      const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kAddV2ParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <>
inline std::uint32_t ParallelForAddV2<std::int64_t>(const CpuKernelContext &ctx, std::int64_t total,
                                                    std::int64_t per_unit_size,
                                                    const std::function<void(std::int64_t, std::int64_t)> &work) {
  const std::int64_t kNumber1 = 32;
  if (total > kNumber1 * 1024)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <>
inline std::uint32_t ParallelForAddV2<std::double_t>(const CpuKernelContext &ctx, std::int64_t total,
                                                     std::int64_t per_unit_size,
                                                     const std::function<void(std::int64_t, std::int64_t)> &work) {
  const std::int64_t kNumber2 = 16;
  if (total > kNumber2 * 1024)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAddV2Kernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *input1{static_cast<T *>(ctx.Input(1)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForAddV2<T>(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input0 + begin, input0 + end, input1 + begin, output + begin, ScalarAddV2<T>);
  });
}

template <typename T>
inline std::uint32_t ComputeAddV2(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAddV2Kernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("AddV2 compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAddV2(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Input(1)->GetDataType()) {
    KERNEL_LOG_ERROR(
      "The data type of the first input [%s] need be the same as the second "
      "input [%s].",
      DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Input(1)->GetDataType()).c_str());
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

inline std::uint32_t CheckAddV2(const CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(const_cast<CpuKernelContext &>(ctx), kAddV2InputNum, kAddV2OutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                                           : ExtraCheckAddV2(ctx);
}

inline std::uint32_t ComputeAddV2(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_INT8:
      return ComputeAddV2<std::int8_t>(ctx);
    case DT_INT16:
      return ComputeAddV2<std::int16_t>(ctx);
    case DT_INT32:
      return ComputeAddV2<std::int32_t>(ctx);
    case DT_INT64:
      return ComputeAddV2<std::int64_t>(ctx);
    case DT_UINT8:
      return ComputeAddV2<std::uint8_t>(ctx);
    case DT_FLOAT16:
      return ComputeAddV2<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAddV2<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAddV2<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeAddV2<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeAddV2<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AddV2CpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAddV2(ctx, kAddV2InputNum, kAddV2OutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                  : detail::ComputeAddV2(ctx);
}

REGISTER_CPU_KERNEL(kAddV2, AddV2CpuKernel);
}  // namespace aicpu
