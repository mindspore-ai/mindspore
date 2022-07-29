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
#ifndef AICPU_OPS_AICPU_RANDOM_CHOICE_WITH_MASK_KERNELS_H_
#define AICPU_OPS_AICPU_RANDOM_CHOICE_WITH_MASK_KERNELS_H_

#include <vector>
#include "common/kernel_base.h"

namespace aicpu {
class RandomChoiceWithMaskKernel : public KernelBase {
 public:
  RandomChoiceWithMaskKernel() : KernelBase("RandomChoiceWithMask") {}
  ~RandomChoiceWithMaskKernel() = default;

 protected:
  int64_t count_ = 0;
  std::vector<int64_t> dims_;
  uint32_t DoCompute() override;
  uint32_t ParseKernelParam() override;
  void UpdateOutputShapeValue(int64_t non_zero_num, int64_t output_length);
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_RANDOM_CHOICE_WITH_MASK_KERNELS_H_
