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

#ifndef MINDSPORE_MATRIX_INVERSE_CPU_KERNEL_H
#define MINDSPORE_MATRIX_INVERSE_CPU_KERNEL_H

#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class MatrixInverseCPUKernel : public CPUKernel {
 public:
  MatrixInverseCPUKernel() = default;
  ~MatrixInverseCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  size_t batch_size_{1};
  size_t size_{1};
  bool adjoint_{false};
};

MS_REG_CPU_KERNEL_T(MatrixInverse, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    MatrixInverseCPUKernel, float);
MS_REG_CPU_KERNEL_T(MatrixInverse, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                    MatrixInverseCPUKernel, double);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MATRIX_INVERSE_CPU_KERNEL_H
