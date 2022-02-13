/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNARY_OP_CPU_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNARY_OP_CPU_KERNEL_H_

#include <complex>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
namespace mindspore {
namespace kernel {
template <typename T, typename S>
class UnaryOpCpuKernelMod : public NativeCpuKernelMod {
 public:
  UnaryOpCpuKernelMod() = default;
  ~UnaryOpCpuKernelMod() override = default;
  using UnaryOpFunc = std::function<void(const T *, S *, size_t, size_t)>;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void GetUnaryOpFunc();
  UnaryOpFunc unary_op_func_{nullptr};
};

MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpCpuKernelMod, complex128, double)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpCpuKernelMod, complex64, float)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      UnaryOpCpuKernelMod, char, char)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      UnaryOpCpuKernelMod, int16_t, int16_t)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpCpuKernelMod, int32_t, int32_t)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      UnaryOpCpuKernelMod, int64_t, int64_t)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      UnaryOpCpuKernelMod, uint16_t, uint16_t)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      UnaryOpCpuKernelMod, uint32_t, uint32_t)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      UnaryOpCpuKernelMod, uint64_t, uint64_t)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpCpuKernelMod, float, float)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpCpuKernelMod, double, double)
MS_REG_CPU_KERNEL_T_S(Real, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      UnaryOpCpuKernelMod, bool, bool)

MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpCpuKernelMod, complex128, double)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpCpuKernelMod, complex64, float)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      UnaryOpCpuKernelMod, char, char)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      UnaryOpCpuKernelMod, int16_t, int16_t)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpCpuKernelMod, int32_t, int32_t)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      UnaryOpCpuKernelMod, int64_t, int64_t)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      UnaryOpCpuKernelMod, uint16_t, uint16_t)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      UnaryOpCpuKernelMod, uint32_t, uint32_t)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      UnaryOpCpuKernelMod, uint64_t, uint64_t)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpCpuKernelMod, float, float)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpCpuKernelMod, double, double)
MS_REG_CPU_KERNEL_T_S(Imag, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      UnaryOpCpuKernelMod, bool, bool)

MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                      UnaryOpCpuKernelMod, complex128, complex128)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                      UnaryOpCpuKernelMod, complex64, complex64)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      UnaryOpCpuKernelMod, char, char)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      UnaryOpCpuKernelMod, int16_t, int16_t)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpCpuKernelMod, int32_t, int32_t)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      UnaryOpCpuKernelMod, int64_t, int64_t)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      UnaryOpCpuKernelMod, uint16_t, uint16_t)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      UnaryOpCpuKernelMod, uint32_t, uint32_t)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      UnaryOpCpuKernelMod, uint64_t, uint64_t)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpCpuKernelMod, float, float)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpCpuKernelMod, double, double)
MS_REG_CPU_KERNEL_T_S(Conj, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      UnaryOpCpuKernelMod, bool, bool)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNARY_OP_CPU_KERNEL_H_
