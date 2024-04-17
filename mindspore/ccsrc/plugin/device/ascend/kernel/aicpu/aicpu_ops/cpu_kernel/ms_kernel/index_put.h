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

#ifndef AICPU_KERNELS_NORMALIZED_INDEX_PUT_H_
#define AICPU_KERNELS_NORMALIZED_INDEX_PUT_H_

#include <vector>
#include "inc/ms_cpu_kernel.h"
#include "cpu_types.h"
#include "utils/bcast.h"

namespace aicpu {
class IndexPutCpuKernel : public CpuKernel {
 public:
  IndexPutCpuKernel() = default;
  ~IndexPutCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t IndexPutParmCheck(CpuKernelContext &ctx);

  void Transpose(std::vector<std::vector<int64_t>> *A) const;

  int64_t Multiplicative(const std::vector<int64_t> &tensorshapes, int64_t start, int64_t end);

  template <typename T>
  bool ComputeNospecial(CpuKernelContext &ctx, std::vector<int64_t> x1_shape, T *x2, size_t x2_nums,
                        std::vector<std::vector<int64_t>> indices_value, T *y, int accumulate);

  template <typename T>
  bool ComputeSpecial(CpuKernelContext &ctx, std::vector<int64_t> x1_shape, T *x2, size_t x2_nums,
                      std::vector<std::vector<int64_t>> indices_value, T *y, int accumulate);

  template <typename T, typename T0>
  uint32_t IndexPutCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_INDEX_PUT_H_
