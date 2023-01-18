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

#ifndef AICPU_KERNELS_NORMALIZED_LOGICALXOR_H_
#define AICPU_KERNELS_NORMALIZED_LOGICALXOR_H_

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {

class LogicalXorCpuKernel : public CpuKernel {
 public:
  LogicalXorCpuKernel() = default;
  ~LogicalXorCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static uint32_t LogicalXorCheck(CpuKernelContext &ctx);

  template <typename T>
  void SpecialCompute(BcastShapeType type, int64_t start, int64_t end, const T *input1, const T *input2, bool *output);

  template <typename T>
  uint32_t NoBcastCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t BcastCompute(CpuKernelContext &ctx, Bcast &bcast);

  template <typename T>
  uint32_t LogicalXorCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
