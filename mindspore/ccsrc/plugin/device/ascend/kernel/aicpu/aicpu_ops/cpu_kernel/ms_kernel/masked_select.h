/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_MASKED_SELECT_H_
#define AICPU_KERNELS_NORMALIZED_MASKED_SELECT_H_

#include <vector>
#include "cpu_kernel/inc/cpu_ops_kernel.h"

namespace aicpu {
class MaskedSelectCpuKernel : public CpuKernel {
 public:
  ~MaskedSelectCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief compute for all types
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T>
  uint32_t MaskedSelectCompute(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t ParallelCompute(const CpuKernelContext &ctx, const std::vector<int64_t> &inputShapeX,
                           const std::vector<int64_t> &inputShapeMask, const std::vector<int64_t> &outputShape,
                           int64_t dataNum);
};
}  // namespace aicpu
#endif
