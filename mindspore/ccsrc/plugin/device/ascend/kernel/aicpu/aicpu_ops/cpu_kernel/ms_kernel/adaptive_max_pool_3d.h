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
#ifndef AICPU_KERNELS_NORMALIZED_ADAPTIVE_MAX_POOL_3D_H_
#define AICPU_KERNELS_NORMALIZED_ADAPTIVE_MAX_POOL_3D_H_

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include <vector>

namespace aicpu {

class AdaptiveMaxPool3dCpuKernel : public CpuKernel {
 public:
  AdaptiveMaxPool3dCpuKernel() = default;
  ~AdaptiveMaxPool3dCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t AdaptiveMaxPool3dCheckAndSetShape(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t AdaptiveMaxPool3dCompute(const CpuKernelContext &ctx);
  int64_t ComputeStride(const std::vector<int64_t> &shape, size_t index);
};
}  // namespace aicpu
#endif
