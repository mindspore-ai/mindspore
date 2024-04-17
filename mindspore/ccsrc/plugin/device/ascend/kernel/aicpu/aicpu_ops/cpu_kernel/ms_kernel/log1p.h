/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_LOG1P_H
#define AICPU_KERNELS_NORMALIZED_LOG1P_H

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class Log1pCpuKernel : public CpuKernel {
 public:
  Log1pCpuKernel() = default;
  ~Log1pCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t Log1pCheck(CpuKernelContext &ctx) const;

  template <typename T>
  uint32_t Log1pCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t Log1pComputeComplex(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
