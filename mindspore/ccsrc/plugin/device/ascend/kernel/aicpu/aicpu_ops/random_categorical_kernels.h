/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_AICPU_OPS_RAMDOM_CATEGORICAL_KERNELS_H_
#define AICPU_AICPU_OPS_RANDOM_CATEGORICAL_KERNELS_H_

#include <random>
#include "common/kernel_base.h"

namespace aicpu {
class RandomCategoricalKernel : public KernelBase {
 public:
  RandomCategoricalKernel() : KernelBase("RandomCategorical"), batch_size_(0), num_classes_(0) {}

  ~RandomCategoricalKernel() = default;

  int64_t batch_size_;
  int num_classes_;
  aicpuops::DataType input_type_;
  aicpuops::DataType output_type_;

 protected:
  uint32_t DoCompute() override;
  template <typename T>
  uint32_t DoComputeWithOutputType(T input_type);
  template <typename T, typename S>
  uint32_t DoComputeForEachType(T input_type, S output_type);
  uint32_t ParseKernelParam() override;

 private:
  std::default_random_engine rng_;
};
}  // namespace aicpu
#endif  // AICPU_AICPU_OPS_RAMDOM_CATEGORICAL_KERNELS_H_
