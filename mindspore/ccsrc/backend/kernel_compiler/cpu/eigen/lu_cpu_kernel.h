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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_LU_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_LU_CPU_KERNEL_H_

#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class LUCPUKernel : public CPUKernel {
 public:
  LUCPUKernel() = default;
  ~LUCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col);
  void InitInputOutputSize(const CNodePtr &kernel_node) override;
  T GetPermutatedValue(const T *lu_value, const std::vector<int> &per_value, size_t i, size_t j);
  bool UpdateMajorPermutation(T *lu_value, std::vector<int> *const per_value, size_t k, size_t rows);
  void SetPermutatedValue(T *lu_value, const std::vector<int> &per_value, size_t i, size_t j, const T &value);
  size_t a_row_{1};
  size_t a_col_{1};
  size_t lu_row_{1};
  size_t lu_col_{1};
  size_t pivots_row_{1};
  size_t pivots_col_{1};
  size_t permutation_row_{1};
  size_t permutation_col_{1};
  TypeId dtype_{kNumberTypeFloat32};
  int *pivots_{nullptr};
};

MS_REG_CPU_KERNEL_T(LU,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    LUCPUKernel, float);
MS_REG_CPU_KERNEL_T(LU,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat64)
                      .AddOutputAttr(kNumberTypeFloat64)
                      .AddOutputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    LUCPUKernel, double);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_LU_CPU_KERNEL_H_
