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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_INVERSE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_INVERSE_CPU_KERNEL_H_
#include <complex>
#include <memory>
#include <unordered_map>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class MatrixInverseCPUKernel : public CPUKernel {
 public:
  MatrixInverseCPUKernel() = default;
  ~MatrixInverseCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  CNodeWeakPtr node_wpt_;
  bool adjoint_{false};
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void LaunchMatrixInverse(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};

MS_REG_CPU_KERNEL(MatrixInverse, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  MatrixInverseCPUKernel);

MS_REG_CPU_KERNEL(MatrixInverse, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  MatrixInverseCPUKernel);

MS_REG_CPU_KERNEL(MatrixInverse, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  MatrixInverseCPUKernel);

MS_REG_CPU_KERNEL(MatrixInverse, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  MatrixInverseCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_INVERSE_CPU_KERNEL_H_
