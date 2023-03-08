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
#ifndef AICPU_KERNELS_GENERATEEODMASK_H_
#define AICPU_KERNELS_GENERATEEODMASK_H_

#include <vector>
#include "common/kernel_base.h"
#include "cpu_ops_kernel.h"

namespace aicpu {
class GenerateEodMaskCpuKernel : public CpuKernel {
 public:
  GenerateEodMaskCpuKernel() = default;
  ~GenerateEodMaskCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t ComputeKernel(CpuKernelContext &ctx, const T &eod_token_id);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_GENERATEEODMASK_H_
