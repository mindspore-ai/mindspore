/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_REDUCE_SUM_H
#define AICPU_KERNELS_NORMALIZED_REDUCE_SUM_H

#include <vector>
#include "inc/ms_cpu_kernel.h"

namespace aicpu {
class ReduceSumCpuKernel : public CpuKernel {
 public:
  ReduceSumCpuKernel() = default;
  ~ReduceSumCpuKernel() override = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ReduceSumCheck(CpuKernelContext &ctx) const;

  template <typename T>
  uint32_t ReduceSumCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t ReduceSumOneAxes(CpuKernelContext &ctx, const T *input_data, std::vector<int64_t> &input_shape,
                            T *output_data, int64_t output_num, std::vector<int64_t> &axes, uint32_t &axes_idx);

  template <typename T, typename T2>
  uint32_t ReduceSumCompute2(CpuKernelContext &ctx);

  template <typename T, typename T2>
  uint32_t ReduceSumOneAxes2(CpuKernelContext &ctx, const T *input_data, int64_t input_num,
                             std::vector<int64_t> input_shape, T *output_data, int64_t output_num,
                             std::vector<int64_t> &axes, uint32_t &axes_idx);

  template <typename T1>
  uint32_t ReduceSumDedupAxes(CpuKernelContext &ctx);

  uint32_t ReduceSumParseAxes(std::vector<int64_t> &input_shape, std::vector<int64_t> &axes, uint32_t &axes_idx,
                              int64_t &inner, int64_t &outer, int64_t &depth) const;

  std::vector<int64_t> axes_;
};
}  // namespace aicpu
#endif
