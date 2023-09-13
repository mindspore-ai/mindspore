/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_OPS_DROP_GEN_MASK_KERNELS_H_
#define AICPU_OPS_DROP_GEN_MASK_KERNELS_H_

#include <random>

#include "common/kernel_base.h"

namespace aicpu {
class DropOutGenMaskKernel : public KernelBase {
 public:
  DropOutGenMaskKernel()
      : KernelBase("DropOutGenMask"),
        seed_(0),
        seed2_(0),
        keep_prob_(0),
        count_(0),
        g_key{0, 0},
        g_offset{0, 0},
        out_(nullptr) {}

  ~DropOutGenMaskKernel() = default;

  uint64_t seed_;
  uint64_t seed2_;
  float keep_prob_;
  uint64_t count_;
  uint64_t g_key[2];
  uint64_t g_offset[2];
  std::default_random_engine rng_;

 protected:
  uint32_t DoCompute() override;

  uint32_t ParseKernelParam() override;

  uint8_t *out_;
};
}  // namespace aicpu
#endif
// #endif  // AICPU_OPS_DROP_GEN_MASK_KERNELS_H_
