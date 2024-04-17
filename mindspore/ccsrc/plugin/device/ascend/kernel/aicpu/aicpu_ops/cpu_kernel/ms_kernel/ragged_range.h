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
#ifndef AICPU_KERNELS_NORMALIZED_RAGGED_RANGE_H_
#define AICPU_KERNELS_NORMALIZED_RAGGED_RANGE_H_

#include "inc/ms_cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {

class RaggedRange : public CpuKernel {
 public:
  RaggedRange() = default;
  ~RaggedRange() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T, typename TSPLITS>
  uint32_t RaggedRangeCompute(const uint32_t nrows, Tensor *starts, Tensor *limits, Tensor *deltas,
                              bool broadcast_starts, bool broadcast_limits, bool broadcast_deltas,
                              Tensor *rt_nested_splits, Tensor *rt_dense_values, CpuKernelContext &ctx);

  template <typename T, typename TSPLITS>
  TSPLITS RangeSize(T start, T limit, T delta);
};
}  // namespace aicpu
#endif
