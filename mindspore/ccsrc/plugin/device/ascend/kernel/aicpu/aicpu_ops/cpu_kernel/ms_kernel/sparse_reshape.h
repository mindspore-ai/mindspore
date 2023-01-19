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
#ifndef AICPU_KERNELS_NORMALIZED_SPARSE_RESHAPE_H_
#define AICPU_KERNELS_NORMALIZED_SPARSE_RESHAPE_H_

#include "cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class SparseReshapeCpuKernel : public CpuKernel {
 public:
  ~SparseReshapeCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  void SpecialCompute(int64_t start, int64_t end, const int64_t *in0, int64_t *out0, const int64_t *input_strides,
                      const int64_t *output_strides, const int64_t input_rank, const int64_t output_rank);
};
}  // namespace aicpu
#endif