/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHOLESKY_CPU_SOLVE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHOLESKY_CPU_SOLVE_KERNEL_H_
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class CholeskySolveCpuKernelMod : public NativeCpuKernelMod {
 public:
  CholeskySolveCpuKernelMod() = default;
  ~CholeskySolveCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitLeftMatrixInfo(const std::vector<size_t> &shape, const bool is_rank_equal, size_t *row, size_t *col);
  void InitRightMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col);

  size_t outer_batch_{1};
  size_t input_a_row_{1};
  size_t input_a_col_{1};
  size_t input_b_row_{1};
  size_t input_b_col_{1};
  size_t output_row_{1};
  size_t output_col_{1};
  TypeId dtype_{kNumberTypeFloat32};
  bool lower_{false};
};

MS_REG_CPU_KERNEL_T(
  CholeskySolve,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  CholeskySolveCpuKernelMod, float)
MS_REG_CPU_KERNEL_T(
  CholeskySolve,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  CholeskySolveCpuKernelMod, double)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CHOLESKY_CPU_SOLVE_KERNEL_H_
