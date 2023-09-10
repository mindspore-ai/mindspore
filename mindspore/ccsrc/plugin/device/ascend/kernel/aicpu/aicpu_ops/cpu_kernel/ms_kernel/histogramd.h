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

#ifndef AICPU_KERNELS_HISTOGRAMD_H_
#define AICPU_KERNELS_HISTOGRAMD_H_

#include "cpu_kernel/inc/cpu_ops_kernel.h"

namespace aicpu {
class HistogramDCpuKernel : public CpuKernel {
 public:
  HistogramDCpuKernel() = default;
  ~HistogramDCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ParamCheck(CpuKernelContext &ctx);

  template <typename T, typename InterType>
  uint32_t DoCompute(const CpuKernelContext &ctx);

  double min_attr = 0.0;
  double max_attr = 0.0;
  int32_t bins = 100;
};
}  // namespace aicpu
#endif
