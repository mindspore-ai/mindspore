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

#ifndef AICPU_KERNELS_NORMALIZED_MUL_H_
#define AICPU_KERNELS_NORMALIZED_MUL_H_
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include <Eigen/Dense>
#include "inc/ms_cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"
#include "inc/ms_cpu_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
class RsqrtGradCpuKernel : public CpuKernel {
 public:
  RsqrtGradCpuKernel() = default;
  ~RsqrtGradCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t RsqrtGradCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t RsqrtGradComputeComplex(CpuKernelContext &ctx);

  template <typename T>
  uint32_t RsqrtGradComputeFP16(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_MUL_H_
