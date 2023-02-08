/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_DENSE_TO_DENSE_SET_OPERATION_H_
#define AICPU_KERNELS_NORMALIZED_DENSE_TO_DENSE_SET_OPERATION_H_

#include <set>
#include "cpu_ops_kernel.h"
namespace aicpu {
enum SetOperation { A_MINUS_B = 0, B_MINUS_A = 1, INTERSECTION = 2, UNION = 3 };

class DenseToDenseSetOperationCpuKernel : public CpuKernel {
 public:
  ~DenseToDenseSetOperationCpuKernel() = default;
  virtual uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t Check(const CpuKernelContext &ctx);
  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);
  template <typename T>
  uint32_t OutputSparseTensor(CpuKernelContext &ctx, const std::vector<int64_t> &output_shape, const int64_t num_values,
                              const std::map<std::vector<int64_t>, std::set<T>> &sets);
  template <typename T>
  void ApplySetOperation(const std::set<T> &set1, const std::set<T> &set2, std::set<T> &result);

  SetOperation set_operation_ = A_MINUS_B;
  bool validate_indices_ = true;
};
}  // namespace aicpu
#endif