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
#ifndef AICPU_KERNELS_NORMALIZED_TRANSPOSE_H_
#define AICPU_KERNELS_NORMALIZED_TRANSPOSE_H_

#include <vector>
#include "cpu_ops_kernel.h"

namespace aicpu {
class TransposeCpuKernel : public CpuKernel {
 public:
  ~TransposeCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  std::vector<int64_t> perm;
  uint32_t TransposeParamCheck(CpuKernelContext &ctx);
  uint32_t GetTransposeValue(Tensor *tensor, std::vector<int64_t> &value);

  template <typename T>
  uint32_t TransposeCompute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  //  AICPU_TRANSPOSE_H