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
#ifndef AICPU_KERNELS_NORMALIZED_SLICE_H_
#define AICPU_KERNELS_NORMALIZED_SLICE_H_

#include "cpu_ops_kernel.h"
#include <vector>

namespace aicpu {
class SliceCpuKernel : public CpuKernel {
 public:
  SliceCpuKernel() = default;
  ~SliceCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  bool is_identity;
  bool slice_dim0;
  std::vector<int64_t> offsets;
  std::vector<int64_t> size;

  uint32_t GetSliceValue(Tensor *tensor, std::vector<int64_t> &value);
  uint32_t SliceCheck(CpuKernelContext &ctx);

  template <typename T>
  uint32_t SliceCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
