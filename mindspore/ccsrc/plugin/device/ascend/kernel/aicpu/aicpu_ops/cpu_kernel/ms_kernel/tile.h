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

#ifndef AICPU_KERNELS_NORMALIZED_TILE_H_
#define AICPU_KERNELS_NORMALIZED_TILE_H_

#include "cpu_ops_kernel.h"

namespace aicpu {
class TileCpuKernel : public CpuKernel {
 public:
  TileCpuKernel() = default;
  ~TileCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T, typename M>
  void CopyMultipleTimes(const T *in_data, int64_t in_size, M multiplier, T *out_data);

  template <typename T, typename M>
  std::pair<int64_t, int64_t> TileOneDimension(const std::vector<int64_t> &in_dimensions, const T *in_data,
                                               const M *multipliers, T *out_data, int64_t dimension);

  template <typename T, typename M>
  uint32_t TileCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif