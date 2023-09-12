/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#ifndef AICPU_KERNELS_NORMALIZED_QUANTILE_H_
#define AICPU_KERNELS_NORMALIZED_QUANTILE_H_

#include <vector>

#include "inc/cpu_ops_kernel.h"
namespace aicpu {
class QuantileCpuKernel : public CpuKernel {
 public:
  QuantileCpuKernel() = default;

  ~QuantileCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t GetInputAndCheck(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t QuantileCompute(const CpuKernelContext &ctx);
  uint32_t MaybeWrapDim(int64_t dim, int64_t dim_post_expr);
  template <typename T>
  void QuantileComputeSerialFunc(int64_t last_shape_size, std::vector<T> *sorted);
  template <typename T>
  void QuantileComputeParallelFunc(size_t start, size_t end, int64_t last_shape_size, std::vector<T> *sorted);

  template <typename T>
  void QuantileComputeDefaultFunc(std::vector<T> *sorted);
  std::vector<int64_t> SetQuantileOutputShape();
  template <typename T>
  void SetOutput(std::vector<int64_t> *out_shape);
  template <typename T>
  uint32_t DoParallelQuantile(const CpuKernelContext &ctx, std::vector<T> sorted, std::vector<int64_t> input_dims);
  int64_t last_shape_size_ = 0;
  bool ignore_nan_ = false;
  bool keep_dims_ = false;
  int dim_ = 0;
  int64_t input_dim_ = 0;
  Tensor *input_ = nullptr;
  Tensor *output_ = nullptr;
  Tensor *q_ = nullptr;
  bool has_nan_ = false;
};
}  // namespace aicpu
#endif
