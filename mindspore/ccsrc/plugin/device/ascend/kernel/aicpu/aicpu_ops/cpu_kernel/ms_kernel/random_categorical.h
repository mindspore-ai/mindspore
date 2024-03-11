/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_RANDOM_CATEGORICAL_H_
#define AICPU_KERNELS_NORMALIZED_RANDOM_CATEGORICAL_H_
#include "inc/ms_cpu_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "random/utils.h"

namespace aicpu {
class RandomCategoricalCpuKernel : public CpuKernel {
 public:
  RandomCategoricalCpuKernel() = default;
  ~RandomCategoricalCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T, typename S>
  uint32_t RandomCategoricalCompute(CpuKernelContext &ctx);

  int64_t batch_size_;
  int num_classes_;
  DataType logits_type_;
  DataType num_sample_type_;
  DataType output_type_;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_RANDOM_CATEGORICAL_H_
