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
#ifndef AICPU_KERNELS_NORMALIZED_INDEX_FILL_H_
#define AICPU_KERNELS_NORMALIZED_INDEX_FILL_H_

#include <vector>

#include "cpu_ops_kernel.h"

namespace aicpu {
class IndexFillCpuKernel : public CpuKernel {
 public:
  ~IndexFillCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  template <typename T>
  void SpecialCompute(int64_t start, int64_t end, const int32_t *input_dim, std::map<int32_t, bool> &index_dict);

  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;
};
}  // namespace aicpu
#endif