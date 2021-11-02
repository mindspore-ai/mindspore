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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGH_CPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGH_CPU_KERNEL_H

#include <vector>
#include <complex>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {

using float_complex = std::complex<float>;
using double_complex = std::complex<double>;

template <typename T, typename C>
class EighCPUKernel : public CPUKernel {
 public:
  EighCPUKernel() = default;
  ~EighCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  size_t m_{1};
  bool compute_eigen_vectors{false};
  TypeId dtype_{kNumberTypeFloat32};
};

MS_REG_CPU_KERNEL_T_S(Eigh,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeComplex64)
                        .AddOutputAttr(kNumberTypeComplex64),
                      EighCPUKernel, float, float_complex);
MS_REG_CPU_KERNEL_T_S(Eigh,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeComplex128)
                        .AddOutputAttr(kNumberTypeComplex128),
                      EighCPUKernel, double, double_complex);

MS_REG_CPU_KERNEL_T_S(Eigh,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeComplex64)
                        .AddOutputAttr(kNumberTypeComplex64),
                      EighCPUKernel, float_complex, float_complex);
MS_REG_CPU_KERNEL_T_S(Eigh,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeComplex128)
                        .AddOutputAttr(kNumberTypeComplex128),
                      EighCPUKernel, double_complex, double_complex);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGH_CPU_KERNEL_H
