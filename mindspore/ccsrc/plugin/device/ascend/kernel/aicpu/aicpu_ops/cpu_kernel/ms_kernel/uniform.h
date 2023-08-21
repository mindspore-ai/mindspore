/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_UNIFORM_H
#define AICPU_KERNELS_NORMALIZED_UNIFORM_H
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include <random>
#include <unordered_map>

#include "cpu_ops_kernel.h"

namespace aicpu {
class UniformCpuKernel : public CpuKernel {
 public:
  UniformCpuKernel() = default;
  ~UniformCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief generate data
   * @param ctx cpu kernel context
   * @param input using to input data
   * @param output using to output data
   */
  template <typename T>
  uint32_t DoCompute(const CpuKernelContext &ctx, Tensor *input, Tensor *output);

  template <typename T>
  uint32_t ParaCompute(const CpuKernelContext &ctx, int64_t input_size, T *outputData, float from, float to);

  template <typename T>
  void UniformCompute(float from, float to, int64_t start, int64_t end, T *outputData);
  uint64_t seed_ = 0;
  uint64_t offset_ = 0;
};
}  // namespace aicpu
#endif  // AICPU_KERNELS_NORMALIZED_UNIFORM_H
