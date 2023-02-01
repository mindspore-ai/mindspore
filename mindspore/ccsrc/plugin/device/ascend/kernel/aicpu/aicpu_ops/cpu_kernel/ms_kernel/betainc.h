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
#ifndef AICPU_KERNELS_NORMALIZED_BETAINC_H
#define AICPU_KERNELS_NORMALIZED_BETAINC_H

#include "cpu_ops_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class BetaincCpuKernel : public CpuKernel {
 public:
  BetaincCpuKernel() = default;
  ~BetaincCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief compute for all types
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T>
  uint32_t BetaincCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_BETAINC_H_
