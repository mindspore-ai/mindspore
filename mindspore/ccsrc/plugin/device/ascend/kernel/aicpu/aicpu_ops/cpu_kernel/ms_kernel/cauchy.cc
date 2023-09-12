/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/cauchy.h"

#include <algorithm>
#include <vector>
#include <random>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kCauchy = "Cauchy";
const int64_t kParallelDataNums = 64 * 1024;
const uint32_t knum = 2;
}  // namespace

#define CAUCHY_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                        \
    uint32_t result = CauchyCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("Cauchy kernel compute failed."); \
      return result;                                     \
    }                                                    \
    break;                                               \
  }
namespace aicpu {
uint32_t CauchyCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_INNER_ERROR, "CauchyCompute check output_tensor is nullptr.");
  auto output_dtype = output_tensor->GetDataType();

  switch (output_dtype) {
    CAUCHY_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    CAUCHY_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      KERNEL_LOG_ERROR("Cauchy kernel data type [%s] not support.", DTypeStr(output_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
};

template <typename T>
uint32_t CauchyCpuKernel::CauchyCompute(const CpuKernelContext &ctx) {
  AttrValue *median = ctx.GetAttr("median");
  if (median != nullptr) {
    median_ = median->GetFloat();
  }

  AttrValue *sigma = ctx.GetAttr("sigma");
  if (sigma != nullptr) {
    sigma_ = sigma->GetFloat();
  }

  Tensor *y_tensor = ctx.Output(0);

  AttrValue *output_size_attr = ctx.GetAttr("size");
  KERNEL_CHECK_NULLPTR(output_size_attr, KERNEL_STATUS_PARAM_INVALID, "CauchyCompute get size failed.");
  std::vector<int64_t> output_size = ctx.GetAttr("size")->GetListInt();
  if (output_size.empty()) {
    KERNEL_LOG_ERROR("CauchyCompute get size is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto y_shape = y_tensor->GetTensorShape();
  y_shape->SetDimSizes(output_size);

  int64_t y_num = y_tensor->NumElements();
  KERNEL_CHECK_NULLPTR(y_tensor->GetData(), KERNEL_STATUS_INNER_ERROR, "CauchyCompute check output_data is nullptr.");
  T *y_data = static_cast<T *>(y_tensor->GetData());
  std::default_random_engine generator(std::random_device{}());

  std::cauchy_distribution<float> cauchy_d(median_, sigma_);
  uint32_t max_core_num = 1;
  if (y_num >= kParallelDataNums) {
    max_core_num = std::max(max_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - knum);
    if (max_core_num > y_num) {
      max_core_num = y_num;
    }
  }

  auto Cauchy_d = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      float data = cauchy_d(generator);
      y_data[i] = static_cast<T>(data);
    }
  };

  uint32_t ret = CpuKernelUtils::ParallelFor(ctx, y_num, y_num / max_core_num, Cauchy_d);
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  KERNEL_LOG_INFO("CauchyCpuKernel::ComputeCauchy end.");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCauchy, CauchyCpuKernel);
}  // namespace aicpu
