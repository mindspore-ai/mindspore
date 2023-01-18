/**
 * Copyright(c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unminimum required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_MAXIMUM_H_
#define AICPU_KERNELS_NORMALIZED_MAXIMUM_H_

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class MaximumCpuKernel : public CpuKernel {
 public:
  MaximumCpuKernel() = default;
  ~MaximumCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t MaximumParamCheck(CpuKernelContext &ctx);

  template <typename T>
  void SpecialCompute(BcastShapeType type, int64_t start, int64_t end, CpuKernelContext &ctx);

  template <typename T>
  void SpecialComputeSameShape(int64_t start, int64_t end, CpuKernelContext &ctx, bool is_float16);

  template <typename T>
  void SpecialComputeXOneElement(int64_t start, int64_t end, CpuKernelContext &ctx, bool is_float16);

  template <typename T>
  void SpecialComputeYOneElement(int64_t start, int64_t end, CpuKernelContext &ctx, bool is_float16);

  template <typename T>
  uint32_t NoBcastCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t BcastCompute(CpuKernelContext &ctx, Bcast &bcast);

  template <typename T>
  void BcastComputeMultiKernel(int64_t start, int64_t end, CpuKernelContext &ctx, Bcast &bcast, bool is_float16);

  template <typename T>
  void BcastComputeOneKernel(CpuKernelContext &ctx, Bcast &bcast, bool is_float16);

  template <typename T>
  uint32_t MaximumCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
