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

#include "cpu_kernel/ms_kernel/add_n.h"
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

#include "context/inc/cpu_kernel_utils.h"
#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "utils/kernel_util.h"
#include "cpu_context.h"

namespace {
const std::uint32_t kAddNInputNum{aicpu::kDynamicInput};
const std::uint32_t kAddNOutputNum{1u};
const char *kAddN{"AddN"};
const std::int64_t kAddNParallelNum{16 * 1024};
}  // namespace

namespace aicpu {
namespace detail {

template <typename T>
inline std::uint32_t ParallelForAddN(CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                     const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kAddNParallelNum)
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  else
    work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAddNKernel(CpuKernelContext &ctx) {
  AttrValue *n_ptr{ctx.GetAttr("N")};
  CUST_KERNEL_CHECK_NULLPTR(ctx, n_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr N failed.");
  std::int64_t per_batch_elements{n_ptr->GetInt()};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Output(0)->NumElements()};
  auto cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelForAddN<T>(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    for (std::int64_t i{begin}; i < end; i++) {
      output[i] = static_cast<T>(0);
      for (std::int64_t j{0}; j < per_batch_elements; j++) {
        output[i] += static_cast<T *>(ctx.Input(j)->GetData())[i];
      }
    }
  });
}

template <typename T>
inline std::uint32_t ComputeAddN(CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAddNKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    CUST_KERNEL_LOG_ERROR(ctx, "AddN compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAddN(CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx, "The data type of the input [%s] need be the same as the output [%s].",
                          DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                          DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *n_ptr = ctx.GetAttr("N");
  CUST_KERNEL_CHECK_NULLPTR(ctx, n_ptr, KERNEL_STATUS_PARAM_INVALID, "Get attr N failed.");
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "The data size of the input [%llu] need be the same as the output "
                          "[%llu].",
                          ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAddN(CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(const_cast<CpuKernelContext &>(ctx), kAddNInputNum, kAddNOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                                         : ExtraCheckAddN(ctx);
}

inline std::uint32_t ComputeAddN(CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_INT8:
      return ComputeAddN<std::int8_t>(ctx);
    case DT_INT16:
      return ComputeAddN<std::int16_t>(ctx);
    case DT_INT32:
      return ComputeAddN<std::int32_t>(ctx);
    case DT_INT64:
      return ComputeAddN<std::int64_t>(ctx);
    case DT_UINT8:
      return ComputeAddN<std::uint8_t>(ctx);
    case DT_UINT16:
      return ComputeAddN<std::uint16_t>(ctx);
    case DT_UINT32:
      return ComputeAddN<std::uint32_t>(ctx);
    case DT_UINT64:
      return ComputeAddN<std::uint64_t>(ctx);
    case DT_FLOAT16:
      return ComputeAddN<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAddN<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAddN<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeAddN<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeAddN<std::complex<std::double_t>>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AddNCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAddN(ctx, kAddNInputNum, kAddNOutputNum) ? KERNEL_STATUS_PARAM_INVALID : detail::ComputeAddN(ctx);
}

REGISTER_MS_CPU_KERNEL(kAddN, AddNCpuKernel);
}  // namespace aicpu
