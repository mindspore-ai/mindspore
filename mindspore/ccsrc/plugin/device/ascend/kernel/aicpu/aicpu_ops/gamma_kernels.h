/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_AICPU_OPS_GAMMA_KERNELS_H_
#define AICPU_AICPU_OPS_GAMMA_KERNELS_H_

#include <vector>
#include "common/kernel_base.h"

namespace aicpu {
class GammaKernel : public KernelBase {
 public:
  GammaKernel() : KernelBase("Gamma") {}

  ~GammaKernel() = default;

 protected:
  uint32_t DoCompute() override;

  uint32_t ParseKernelParam() override;

  float *alpha_ = NULL;
  float *beta_ = NULL;
  int64_t *seed_ = NULL;
  int64_t *seed2_ = NULL;
  std::vector<uint64_t> shape;
  std::vector<uint64_t> alpha_shape;
  std::vector<uint64_t> beta_shape;
  std::vector<uint64_t> out_shape;
  uint64_t alpha_count_ = 1;
  uint64_t beta_count_ = 1;
  uint64_t out_count_ = 1;
};
}  // namespace aicpu
#endif  // AICPU_AICPU_OPS_GAMMA_KERNELS_H_
