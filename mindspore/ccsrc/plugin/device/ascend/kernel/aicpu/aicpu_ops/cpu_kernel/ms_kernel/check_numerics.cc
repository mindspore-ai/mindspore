/**
 * Copyright 2021 Jilin University
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved..
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

#include "cpu_kernel/ms_kernel/check_numerics.h"

#include "securec/include/securec.h"

#include <securec.h>
#include <algorithm>
#include "unsupported/Eigen/CXX11/Tensor"

#include "common/kernel_log.h"
#include "common/status.h"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "cpu_kernel/inc/cpu_types.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kCheckNumericsInputNum{1};
const std::uint32_t kCheckNumericsOutputNum{1};
const char *const kCheckNumerics{"CheckNumerics"};
const std::int64_t kCheckNumericsParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline bool ScalarCheckNumerics(const T x) {
  return !std::isfinite(x);
}
template <>
inline bool ScalarCheckNumerics(const Eigen::half x) {
  return !Eigen::half_impl::isfinite(x);
}
inline std::uint32_t ParallelForCheckNumerics(const CpuKernelContext &ctx, std::int64_t total,
                                              std::int64_t per_unit_size,
                                              const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kCheckNumericsParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}
template <typename T>
inline std::uint32_t ComputeCheckNumericsKernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t core_num{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, core_num - 2L), total)};
  bool flag = false;
  std::uint32_t ret = ParallelForCheckNumerics(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    flag = flag || std::any_of(input0 + begin, input0 + end, ScalarCheckNumerics<T>);
    if (!flag) {
      auto ret = memcpy_s(output + begin, static_cast<size_t>((end - begin) * sizeof(T)), input0 + begin,
                          static_cast<size_t>((end - begin) * sizeof(T)));
      if (ret != EOK) {
        KERNEL_LOG_ERROR("memcpy_s error");
      }
    }
  });
  return flag ? KERNEL_STATUS_PARAM_INVALID : ret;
}
template <typename T>
inline std::uint32_t ComputeCheckNumerics(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeCheckNumericsKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CheckNumerics compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckCheckNumerics(const CpuKernelContext &ctx) {
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

inline std::uint32_t ComputeCheckNumerics(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeCheckNumerics<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeCheckNumerics<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeCheckNumerics<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t CheckNumericsCpuKernel::Compute(CpuKernelContext &ctx) {
  std::uint32_t check = NormalCheck(ctx, kCheckNumericsInputNum, kCheckNumericsOutputNum)
                          ? KERNEL_STATUS_PARAM_INVALID
                          : detail::ExtraCheckCheckNumerics(ctx);
  return check ? KERNEL_STATUS_PARAM_INVALID : detail::ComputeCheckNumerics(ctx);
}

REGISTER_CPU_KERNEL(kCheckNumerics, CheckNumericsCpuKernel);
}  // namespace aicpu
