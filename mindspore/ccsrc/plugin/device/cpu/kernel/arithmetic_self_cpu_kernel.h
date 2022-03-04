/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_

#include <complex>
#include <memory>
#include <set>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class ArithmeticSelfCpuKernelMod : public NativeCpuKernelMod {
 public:
  ArithmeticSelfCpuKernelMod() = default;
  ~ArithmeticSelfCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  void LaunchLogicalNot(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T>
  void LaunchKernelComplex(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  TypeId dtype_{kTypeUnknown};
};

template <typename T>
class IdentityCpuKernelMod : public ArithmeticSelfCpuKernelMod {
 public:
  IdentityCpuKernelMod() = default;
  ~IdentityCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
};

MS_REG_CPU_KERNEL(Square, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Square, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Square, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Square, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Square, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Square, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Neg, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sign, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sign, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sign, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Floor, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Floor, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Rint, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Rint, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Round, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Round, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Reciprocal, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Reciprocal, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(GeLU, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(LogicalNot, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Asin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Asin, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(ACos, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(ACos, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Atan, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Atan, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sin, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cos, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cos, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sin, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cos, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sin, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cos, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Tan, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Tan, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sinh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sinh, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cosh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cosh, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sinh, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cosh, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sinh, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Cosh, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Asinh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Asinh, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Acosh, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Acosh, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Asinh, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Asinh, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Acosh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Acosh, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Atanh, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Atanh, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Abs, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Abs, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Abs, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Abs, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);
MS_REG_CPU_KERNEL(Sqrt, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                  ArithmeticSelfCpuKernelMod);

MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                    IdentityCpuKernelMod, uint64_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    IdentityCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                    IdentityCpuKernelMod, uint32_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    IdentityCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                    IdentityCpuKernelMod, uint16_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                    IdentityCpuKernelMod, int16_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                    IdentityCpuKernelMod, uint8_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                    IdentityCpuKernelMod, int8_t);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                    IdentityCpuKernelMod, complex64);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                    IdentityCpuKernelMod, complex128);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                    IdentityCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    IdentityCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                    IdentityCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(Identity, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                    IdentityCpuKernelMod, bool);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_SELF_CPU_KERNEL_H_
