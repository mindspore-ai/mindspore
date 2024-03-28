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

#ifndef AICPU_KERNELS_NORMALIZED_BESSEL_I0_H_
#define AICPU_KERNELS_NORMALIZED_BESSEL_I0_H_

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class BesselI0CpuKernel : public CpuKernel {
 public:
  BesselI0CpuKernel() = default;
  ~BesselI0CpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParallelForCompute(CpuKernelContext &ctx);

  template <typename T>
  void BesselI0Compute(int64_t start, int64_t end, CpuKernelContext &ctx);

  void BesselI0ComputeFloat16(int64_t start, int64_t end, CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_BESSEL_I0_H_
