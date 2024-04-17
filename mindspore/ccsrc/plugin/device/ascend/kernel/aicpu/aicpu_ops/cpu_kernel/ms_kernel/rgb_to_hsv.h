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

#ifndef AICPU_KERNELS_NORMALIZED_RGBToHSV_H_
#define AICPU_KERNELS_NORMALIZED_RGBToHSV_H_

#include <map>
#include <string>
#include <vector>

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class RGBToHSVCpuKernel : public CpuKernel {
 public:
  RGBToHSVCpuKernel() = default;

  ~RGBToHSVCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename TInput, typename TOutput>
  static uint32_t DoCompute(CpuKernelContext &ctx);

  uint32_t CheckParams(CpuKernelContext &ctx);

  uint32_t CheckParam(CpuKernelContext &ctx, const std::string &in_or_out, uint32_t index, size_t rank);

  uint32_t CheckShapes(CpuKernelContext &ctx);

 private:
  using KernelFunction = uint32_t (*)(CpuKernelContext &ctx);
  static const std::map<std::string, KernelFunction> kernels_;
  static const std::vector<std::string> kernels_name_;
};
}  // namespace aicpu
#endif
