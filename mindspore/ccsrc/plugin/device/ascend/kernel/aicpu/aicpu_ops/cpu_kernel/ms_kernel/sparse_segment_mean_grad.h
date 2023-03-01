/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSE_SEGMENT_MEAN_GRAD_H_
#define AICPU_KERNELS_NORMALIZED_SPARSE_SEGMENT_MEAN_GRAD_H_

#include "cpu_ops_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class SparseSegmentMeanGradCpuKernel : public CpuKernel {
 public:
  SparseSegmentMeanGradCpuKernel() = default;
  ~SparseSegmentMeanGradCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  KernelStatus CheckDataPara(const CpuKernelContext &ctx) const;
  KernelStatus CheckShapePara(const CpuKernelContext &ctx) const;
  template <typename T>
  KernelStatus ComputeKernel(const CpuKernelContext &ctx);

  template <typename T, typename T1, typename T2>
  KernelStatus ComputeKernelWithType(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif