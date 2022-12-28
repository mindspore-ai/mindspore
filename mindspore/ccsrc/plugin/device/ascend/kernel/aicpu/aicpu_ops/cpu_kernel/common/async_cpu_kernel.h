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

#ifndef ASYNC_CPU_KERNEL_H
#define ASYNC_CPU_KERNEL_H

#include "cpu_kernel/inc/cpu_ops_kernel.h"

namespace aicpu {
class AICPU_VISIBILITY AsyncCpuKernel : public CpuKernel {
 public:
  using CpuKernel::CpuKernel;

  using DoneCallback = std::function<void(uint32_t status)>;

  virtual uint32_t ComputeAsync(CpuKernelContext &ctx, DoneCallback done) = 0;

  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
#endif  // ASYNC_CPU_KERNEL_H