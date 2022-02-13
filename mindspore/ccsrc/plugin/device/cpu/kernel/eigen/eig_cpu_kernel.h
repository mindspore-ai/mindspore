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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIG_CPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIG_CPU_KERNEL_H

#include <vector>
#include <complex>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {

using float_complex = std::complex<float>;
using double_complex = std::complex<double>;

/**
 * this is for Generic matrix eigenvalues and eigenvectors
 * @tparam T , input Type
 * @tparam C , output Type, complex
 */
template <typename T, typename C>
class EigCpuKernelMod : public NativeCpuKernelMod {
 public:
  EigCpuKernelMod() = default;
  ~EigCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitMatrixInfo(const std::vector<size_t> &shape);
  bool compute_v_{true};
  size_t row_size_{1};
  size_t col_size_{1};
  size_t batch_size_{1};
};

// If compute_v is false.
MS_REG_CPU_KERNEL_T_S(Eig, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
                      EigCpuKernelMod, float, float_complex);
MS_REG_CPU_KERNEL_T_S(Eig, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
                      EigCpuKernelMod, double, double_complex);
MS_REG_CPU_KERNEL_T_S(Eig, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                      EigCpuKernelMod, float_complex, float_complex);
MS_REG_CPU_KERNEL_T_S(Eig, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                      EigCpuKernelMod, double_complex, double_complex);
// If compute_v is true.
MS_REG_CPU_KERNEL_T_S(
  Eig,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
  EigCpuKernelMod, float, float_complex);
MS_REG_CPU_KERNEL_T_S(Eig,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddOutputAttr(kNumberTypeComplex128)
                        .AddOutputAttr(kNumberTypeComplex128),
                      EigCpuKernelMod, double, double_complex);
MS_REG_CPU_KERNEL_T_S(Eig,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex64)
                        .AddOutputAttr(kNumberTypeComplex64)
                        .AddOutputAttr(kNumberTypeComplex64),
                      EigCpuKernelMod, float_complex, float_complex);
MS_REG_CPU_KERNEL_T_S(Eig,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeComplex128)
                        .AddOutputAttr(kNumberTypeComplex128)
                        .AddOutputAttr(kNumberTypeComplex128),
                      EigCpuKernelMod, double_complex, double_complex);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_EIG_CPU_KERNEL_H
