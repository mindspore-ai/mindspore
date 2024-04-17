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

#include "cpu_kernel/ms_kernel/cos.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>

#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kCosInputNum{1u};
const std::uint32_t kCosOutputNum{1u};
const char *kCos{"Cos"};
const std::int64_t kCosParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarCos(T x) {
  return std::cos(x);
}

template <>
inline Eigen::half ScalarCos(Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(std::cos(static_cast<std::float_t>(x)))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}

inline std::uint32_t ParallelForCos(CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                    const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kCosParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeCosKernel(CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForCos(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input0 + begin, input0 + end, output + begin, ScalarCos<T>);
  });
}

template <typename T>
inline std::uint32_t ComputeCos(CpuKernelContext &ctx) {
  std::uint32_t result{ComputeCosKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "Cos compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckCos(CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data type of the input [%s] need be the same as the output [%s].",
                          DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                          DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data size of the input [%llu] need be the same as the output "
                          "[%llu].",
                          ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t ComputeCos(CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeCos<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeCos<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeCos<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeCos<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeCos<std::complex<std::double_t>>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t CosCpuKernel::Compute(CpuKernelContext &ctx) {
  std::uint32_t check = NormalCheck(const_cast<CpuKernelContext &>(ctx), kCosInputNum, kCosOutputNum)
                          ? KERNEL_STATUS_PARAM_INVALID
                          : detail::ExtraCheckCos(ctx);
  return check ? KERNEL_STATUS_PARAM_INVALID : detail::ComputeCos(ctx);
}

REGISTER_MS_CPU_KERNEL(kCos, CosCpuKernel);
}  // namespace aicpu
