/**
 * Copyright 2021 Huawei Technologies Co., Ltd.
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

#include "sqrt.h"

#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kSqrtInputNum{1};
const std::uint32_t kSqrtOutputNum{1};
const std::uint32_t Parallel4ThreadNum{4096};
const std::uint32_t Parallel6ThreadNum{8192};
const std::uint32_t ParallelNum{16384};
const char *kSqrt{"Sqrt"};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeSqrtKernel(const CpuKernelContext &ctx) {
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  std::int64_t total = ctx.Input(0)->NumElements();
  std::uint64_t total_size = ctx.Input(0)->GetDataSize();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  bool parallel_flag = false;
  if (total_size > ParallelNum * sizeof(T)) {
    parallel_flag = true;
  } else if (total_size > Parallel6ThreadNum * sizeof(T)) {
    parallel_flag = true;
    cores = 8;
  } else if (total_size > Parallel4ThreadNum * sizeof(T)) {
    parallel_flag = true;
    cores = 6;
  }
  if (parallel_flag) {
    std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
    return ParallelFor(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
      std::transform(input + begin, input + end, output + begin, Eigen::numext::sqrt<T>);
    });
  } else if (cores != 0) {
    std::transform(input, input + total, output, Eigen::numext::sqrt<T>);
  } else {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeSqrt(const CpuKernelContext &ctx) {
  uint32_t result = ComputeSqrtKernel<T>(ctx);
  if (result != 0) {
    KERNEL_LOG_ERROR("Sqrt compute failed.");
  }
  return result;
}

inline std::uint32_t SqrtExtraCheck(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    KERNEL_LOG_ERROR(
      "The data dim of the input size [%llu] need be the same as the output "
      "size [%llu].",
      input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != output_dims[index]) {
      KERNEL_LOG_ERROR(
        "The data dim[%llu]=%lld of the input need be the same as the output "
        "dim[%llu]=%lld.",
        index, input_dims[index], index, output_dims[index]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

std::uint32_t SqrtCheck(CpuKernelContext &ctx, uint32_t inputs_num, uint32_t outputs_num) {
  return NormalCheck(ctx, kSqrtInputNum, kSqrtOutputNum) ? KERNEL_STATUS_PARAM_INVALID : SqrtExtraCheck(ctx);
}

std::uint32_t SqrtCompute(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeSqrt<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeSqrt<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeSqrt<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeSqrt<std::complex<std::float_t> >(ctx);
    case DT_COMPLEX128:
      return ComputeSqrt<std::complex<std::double_t> >(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t SqrtCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::SqrtCheck(ctx, kSqrtInputNum, kSqrtOutputNum) ? KERNEL_STATUS_PARAM_INVALID : detail::SqrtCompute(ctx);
}

REGISTER_CPU_KERNEL(kSqrt, SqrtCpuKernel);
}  // namespace aicpu
