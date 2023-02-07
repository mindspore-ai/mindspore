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

#ifndef AICPU_KERNELS_NORMALIZED_SEGMENT_MIN_H_
#define AICPU_KERNELS_NORMALIZED_SEGMENT_MIN_H_

#include "cpu_ops_kernel.h"

namespace aicpu {
class SegmentMinCpuKernel : public CpuKernel {
 public:
  SegmentMinCpuKernel() = default;
  ~SegmentMinCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <class T1, class T2>
  static uint32_t SegmentMinCompute(CpuKernelContext &ctx);
  static uint32_t SegmentMinCheck(CpuKernelContext &ctx);
  static bool CheckType(Tensor *t);
  static bool CheckDim(Tensor *t);
  static bool CheckSorted(Tensor *t);
  static bool CheckLength(Tensor *seg, Tensor *data);
};
}  // namespace aicpu
#endif
