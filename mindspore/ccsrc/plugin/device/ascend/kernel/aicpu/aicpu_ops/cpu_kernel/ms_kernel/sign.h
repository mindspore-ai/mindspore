/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_SIGN_H
#define AICPU_KERNELS_NORMALIZED_SIGN_H

#include "inc/ms_cpu_kernel.h"
#include "context/common/status.h"

namespace aicpu {
class SignCpuKernel : public CpuKernel {
 public:
  SignCpuKernel() = default;
  ~SignCpuKernel() override = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  KernelStatus SignCheck(CpuKernelContext &ctx) const;

  template <typename T>
  uint32_t SignCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SignComputeComplex(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
