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
#ifndef AICPU_KERNELS_NORMALIZED_ROUND_H_
#define AICPU_KERNELS_NORMALIZED_ROUND_H_

#include "cpu_ops_kernel.h"
#include <unordered_set>
#include <string>
#include "utils/sparse_tensor.h"

namespace aicpu {
class SetSizeCpuKernel : public CpuKernel {
 public:
  SetSizeCpuKernel() = default;
  ~SetSizeCpuKernel() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t SetSizeCompute(CpuKernelContext &ctx, SparseTensor &st);
  uint32_t SetSizeCompute_string(CpuKernelContext &ctx, SparseTensor &st);
  uint32_t SparseTensorFromContext(CpuKernelContext &ctx, const bool validate_indices, SparseTensor &st);
  template <typename T>
  uint32_t PopulateFromSparseGroup(CpuKernelContext &ctx, const Group &group,
                                   const std::vector<int64_t> &sparse_tensor_shape, std::unordered_set<T> *result);
  template <typename T>
  uint32_t CheckGroup(CpuKernelContext &ctx, const Group &group, const std::vector<int64_t> &sparse_tensor_shape);
  bool validate_indices_ = true;
  uint32_t IndicesValid(CpuKernelContext &ctx, SparseTensor &st);

  int32_t dims_;
  std::unordered_set<std::string> all_indices_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> order_;
  Tensor *set_indices_ = nullptr;
  Tensor *set_values_ = nullptr;
  Tensor *set_shape_ = nullptr;
  Tensor *output_ = nullptr;
};
}  // namespace aicpu
#endif