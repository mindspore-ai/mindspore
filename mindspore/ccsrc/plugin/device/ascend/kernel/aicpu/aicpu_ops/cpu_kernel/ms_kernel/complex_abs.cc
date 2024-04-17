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

#include "cpu_kernel/ms_kernel/complex_abs.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kComplexAbsInputNum{1u};
const std::uint32_t kComplexAbsOutputNum{1u};
const char *kComplexAbs{"ComplexAbs"};
const std::int64_t kComplexAbsParallelNum{32 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline const typename T::value_type ScalarComplexAbs(const T &x) {
  return std::abs(x);
}
inline std::uint32_t ParallelForComplexAbs(CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                           const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kComplexAbsParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}
template <typename T>
inline std::uint32_t ComputeComplexAbsKernel(CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  typename T::value_type *output{static_cast<typename T::value_type *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForComplexAbs(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input0 + begin, input0 + end, output + begin, ScalarComplexAbs<T>);
  });
}

template <typename T>
inline std::uint32_t ComputeComplexAbs(CpuKernelContext &ctx) {
  std::uint32_t result{ComputeComplexAbsKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "ComplexAbs compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckComplexAbs(CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() == ctx.Output(0)->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data type of the input [%s] should not be the same as the output "
                          "[%s].",
                          DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                          DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize() * 2) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data size of the input [%llu] need be as twice as the output "
                          "[%llu].",
                          ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t ComputeComplexAbs(CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_COMPLEX64:
      return ComputeComplexAbs<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeComplexAbs<std::complex<std::double_t>>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t ComplexAbsCpuKernel::Compute(CpuKernelContext &ctx) {
  std::uint32_t check = NormalCheck(ctx, kComplexAbsInputNum, kComplexAbsOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                                    : detail::ExtraCheckComplexAbs(ctx);
  return check ? KERNEL_STATUS_PARAM_INVALID : detail::ComputeComplexAbs(ctx);
}

REGISTER_MS_CPU_KERNEL(kComplexAbs, ComplexAbsCpuKernel);
}  // namespace aicpu
