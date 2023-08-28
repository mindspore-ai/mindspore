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

#include "cpu_kernel/ms_kernel/adjust_contrastv2.h"
#include <algorithm>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "common/kernel_log.h"
#include "cpu_kernel/common/status.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/inc/cpu_context.h"

namespace {
const std::uint32_t kAdjustContrastv2InputNum{2u};
const std::uint32_t kAdjustContrastv2OutputNum{1u};
const char *kAdjustContrastv2{"AdjustContrastv2"};
const std::int64_t kAdjustContrastv2ParallelNum{64 * 1024};
const std::uint64_t kFloatSize = 4;
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline void AdjustContrastv2(T *image, T *image_out, std::float_t contrast_factor, std::int64_t channel_count,
                             std::int64_t per_batch_elements) {
  for (std::int64_t j = 0; j < channel_count; j++) {
    std::float_t sum{0.0f};
    for (std::int64_t i = 0; i < per_batch_elements; i += channel_count) {
      sum += static_cast<std::float_t>(image[i + j]);
    }
    if (per_batch_elements == 0 || channel_count == 0) {
      KERNEL_LOG_ERROR("per_batch_elements or channel_count is 0.");
      return;
    }
    std::float_t mean{sum / (per_batch_elements / channel_count)};
    for (std::int64_t i = 0; i < per_batch_elements; i += channel_count) {
      image_out[i + j] = static_cast<T>((static_cast<std::float_t>(image[i + j]) - mean) * contrast_factor + mean);
    }
  }
}

inline std::uint32_t ParallelForAdjustContrastv2(const CpuKernelContext &ctx, std::int64_t total,
                                                 std::int64_t per_unit_size,
                                                 const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kAdjustContrastv2ParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAdjustContrastv2Kernel(const CpuKernelContext &ctx) {
  T *input{static_cast<T *>(ctx.Input(0)->GetData())};
  std::float_t *contrast_factor{static_cast<std::float_t *>(ctx.Input(1)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  const std::vector<std::int64_t> &x_dim_sizes{ctx.Output(0)->GetTensorShape()->GetDimSizes()};
  std::size_t n{x_dim_sizes.size()};
  std::int64_t per_batch_elements{x_dim_sizes[n - 1] * x_dim_sizes[n - 2] * x_dim_sizes[n - 3]};
  std::int64_t total{ctx.Input(0)->NumElements() / per_batch_elements};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForAdjustContrastv2(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i = begin; i < end; i++) {
      AdjustContrastv2(&(input[i * per_batch_elements]), &(output[i * per_batch_elements]), contrast_factor[0],
                       x_dim_sizes[n - 1], per_batch_elements);
    }
  });
}

template <typename T>
inline std::uint32_t ComputeAdjustContrastv2(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAdjustContrastv2Kernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("AdjustContrastv2 compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAdjustContrastv2(const CpuKernelContext &ctx) {
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
  if (ctx.Input(1)->GetDataType() != aicpu::DataType::DT_FLOAT) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be [%s].", DTypeStr(ctx.Input(1)->GetDataType()).c_str(),
                     DTypeStr(aicpu::DataType::DT_FLOAT).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(1)->GetDataSize() != kFloatSize) {
    KERNEL_LOG_ERROR("The data size of the input [%llu] need be [%llu].", ctx.Input(1)->GetDataSize(), kFloatSize);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAdjustContrastv2(const CpuKernelContext &ctx, std::uint32_t inputs_num,
                                           std::uint32_t outputs_num) {
  return NormalCheck(const_cast<CpuKernelContext &>(ctx), kAdjustContrastv2InputNum, kAdjustContrastv2OutputNum)
           ? KERNEL_STATUS_PARAM_INVALID
           : ExtraCheckAdjustContrastv2(ctx);
}

inline std::uint32_t ComputeAdjustContrastv2(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAdjustContrastv2<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAdjustContrastv2<std::float_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AdjustContrastv2CpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAdjustContrastv2(ctx, kAdjustContrastv2InputNum, kAdjustContrastv2OutputNum)
           ? KERNEL_STATUS_PARAM_INVALID
           : detail::ComputeAdjustContrastv2(ctx);
}

REGISTER_CPU_KERNEL(kAdjustContrastv2, AdjustContrastv2CpuKernel);
}  // namespace aicpu
