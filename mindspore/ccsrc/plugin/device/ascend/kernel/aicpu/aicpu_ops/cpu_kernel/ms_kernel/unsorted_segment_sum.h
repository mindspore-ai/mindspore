/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_UNSORTED_SEGMENT_SUM_H
#define AICPU_KERNELS_NORMALIZED_UNSORTED_SEGMENT_SUM_H

#include "cpu_ops_kernel.h"

namespace aicpu {

class UnsortedSegmentSumCpuKernel : public CpuKernel {
 public:
  ~UnsortedSegmentSumCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename input_t, typename segment_ids_t, typename num_segments_t>
  uint32_t UnsortedSegmentSumComputeTemplate(CpuKernelContext &ctx);
  template <typename input_t, typename segment_ids_t>
  uint32_t DoComputeWithNumSegmentsType(CpuKernelContext &ctx, DataType num_segments_type);
  template <typename input_t>
  uint32_t DoComputeWithSegmentIdsType(CpuKernelContext &ctx, DataType segment_ids_type);
};
}  // namespace aicpu
#endif