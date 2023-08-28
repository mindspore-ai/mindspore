/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_BATCHNORMGRADGRAD_H_
#define AICPU_KERNELS_NORMALIZED_BATCHNORMGRADGRAD_H_

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "utils/bcast.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
class BatchNormGradGradCpuKernel : public CpuKernel {
 public:
  BatchNormGradGradCpuKernel() = default;
  ~BatchNormGradGradCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t ParallelCompute(const CpuKernelContext &ctx);

  template <typename T>
  void TrainingComputeNHWC(const CpuKernelContext &ctx, int start, int end);

  template <typename T>
  void InferenceComputeNHWC(const CpuKernelContext &ctx, int start, int end);

  template <typename T>
  void TrainingComputeNCHW(const CpuKernelContext &ctx);

  template <typename T>
  void InferenceComputeNCHW(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
