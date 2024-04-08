/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_LEFT_SHIFT_H_
#define AICPU_KERNELS_NORMALIZED_LEFT_SHIFT_H_

#include "inc/ms_cpu_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class LeftShiftCpuKernel : public CpuKernel {
 public:
  LeftShiftCpuKernel() = default;
  ~LeftShiftCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t LeftShiftParamCheck(CpuKernelContext &ctx);

  template <typename T>
  void SpecialCompute(BcastShapeType type, int64_t start, int64_t end, CpuKernelContext &ctx);

  template <typename T>
  uint32_t NoBcastCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t BcastCompute(CpuKernelContext &ctx, const Bcast &bcast);

  template <typename T>
  uint32_t LeftShiftCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
