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

#include "cpu_kernel/ms_kernel/sinh.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <complex>
#include <vector>
#include "inc/kernel_log.h"
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "frontend/parallel/status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kSinhInputNum{1};
const std::uint32_t kSinhOutputNum{1};
const std::uint32_t ParallelNum{4096};
const char *kSinh{"Sinh"};
}  // namespace

namespace internal {
template <typename T>
inline T ScalarSinh(T x) {
  return Eigen::numext::sinh(x);
}

template <>
inline Eigen::half ScalarSinh(Eigen::half x) {
  const Eigen::half val{Eigen::numext::sinh(static_cast<float>(x))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}
}  // namespace internal

namespace aicpu {
namespace detail {
template <typename T>
inline std::uint32_t ComputeSinhKernel(CpuKernelContext &ctx) {
  const auto ParallelFor = aicpu::CpuKernelUtils::ParallelFor;
  const auto ScalarSinh = internal::ScalarSinh<T>;
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  std::int64_t total = ctx.Input(0)->NumElements();
  std::uint64_t total_size = ctx.Input(0)->GetDataSize();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  if (total_size > ParallelNum * sizeof(T)) {
    std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
    return ParallelFor(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
      std::transform(input + begin, input + end, output + begin, ScalarSinh);
    });
  } else if (cores != 0) {
    std::transform(input, input + total, output, ScalarSinh);
  } else {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeSinh(CpuKernelContext &ctx) {
  uint32_t result = ComputeSinhKernel<T>(ctx);
  if (result != 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "Sinh compute failed.");
  }
  return result;
}

inline std::uint32_t SinhExtraCheck(CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data type of the input [%s] need be the same as the     output [%s].",
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
  std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data dim size of the input [%llu] need be the same as the output "
                          "[%llu].",
                          input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t index = 0; index < input_dims.size(); index++) {
    if (input_dims[index] != output_dims[index]) {
      CUST_KERNEL_LOG_ERROR(ctx, "The data dim of the input need be the same as the output.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

std::uint32_t SinhCheck(CpuKernelContext &ctx, uint32_t inputs_num, uint32_t outputs_num) {
  return NormalCheck(ctx, kSinhInputNum, kSinhOutputNum) ? KERNEL_STATUS_PARAM_INVALID : SinhExtraCheck(ctx);
}

std::uint32_t SinhCompute(CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeSinh<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeSinh<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeSinh<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeSinh<std::complex<std::float_t> >(ctx);
    case DT_COMPLEX128:
      return ComputeSinh<std::complex<std::double_t> >(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t SinhCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::SinhCheck(ctx, kSinhInputNum, kSinhOutputNum) ? KERNEL_STATUS_PARAM_INVALID : detail::SinhCompute(ctx);
}

REGISTER_MS_CPU_KERNEL(kSinh, SinhCpuKernel);
}  // namespace aicpu
