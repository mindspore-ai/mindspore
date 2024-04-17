/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef AICPU_AICPU_OPS_POISSON_KERNELS_H_
#define AICPU_AICPU_OPS_POISSON_KERNELS_H_

#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class PoissonKernel : public CpuKernel {
 public:
  PoissonKernel() = default;
  ~PoissonKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParseKernelParam(CpuKernelContext &ctx);

  float *mean_ = NULL;
  int64_t *seed_ = NULL;
  int64_t *seed2_ = NULL;
  std::vector<uint64_t> shape;
  std::vector<uint64_t> mean_shape;
  std::vector<uint64_t> out_shape;
  uint64_t mean_count_ = 1;
  uint64_t out_count_ = 1;
};
}  // namespace aicpu
#endif  // AICPU_AICPU_OPS_POISSON_KERNELS_H_
