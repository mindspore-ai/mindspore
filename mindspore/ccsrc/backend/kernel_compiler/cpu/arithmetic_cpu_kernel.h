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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_CPU_KERNEL_H_

#include <vector>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/nnacl/arithmetic.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ArithmeticCpuKernelMod : public NativeCpuKernelMod {
 public:
  ArithmeticCpuKernelMod() = default;
  ~ArithmeticCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitComputeFunc();
  void Sub(const T *input1, const T *input2, T *out);
  void Add(const T *input1, const T *input2, T *out);
  void Mul(const T *input1, const T *input2, T *out);
  void RealDiv(const T *input1, const T *input2, T *out);
  void Div(const T *input1, const T *input2, T *out);
  void FloorDiv(const T *input1, const T *input2, T *out);
  void Mod(const T *input1, const T *input2, T *out);
  void FloorMod(const T *input1, const T *input2, T *out);
  void Pow(const T *input1, const T *input2, T *out);
  void AssignAdd(T *input1, const T *input2, T *out);
  void Atan2(const T *input1, const T *input2, T *out);
  void SquaredDifference(const T *input1, const T *input2, T *out);

  using TypeComputeFunc = std::function<void(ArithmeticCpuKernelMod *, const T *in_x, const T *in_y, T *out)>;
  TypeComputeFunc compute_func_{nullptr};
  size_t output_size_{1};
  ArithmeticParameter op_para_{};

  std::vector<size_t> input_shape1_;
  std::vector<size_t> input_shape2_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> input_element_num2_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_element_num_;
};

MS_REG_CPU_KERNEL_T(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  Sub, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  Sub, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  Mul, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  Mul, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  Div, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  Div, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  Div, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  Pow, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  Pow, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  Pow, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  RealDiv, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ArithmeticCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  RealDiv, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  RealDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  ArithmeticCpuKernelMod, int8_t);
MS_REG_CPU_KERNEL_T(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int);
MS_REG_CPU_KERNEL_T(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ArithmeticCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  FloorDiv, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  ArithmeticCpuKernelMod, uint8_t);
MS_REG_CPU_KERNEL_T(
  FloorDiv,
  KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
  ArithmeticCpuKernelMod, uint16_t);
MS_REG_CPU_KERNEL_T(
  Mod, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int);
MS_REG_CPU_KERNEL_T(
  Mod, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  Mod, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  FloorMod, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  FloorMod, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(
  FloorMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  FloorMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ArithmeticCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(
  AssignAdd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int32_t);
MS_REG_CPU_KERNEL_T(
  AssignAdd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  AssignAdd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  ArithmeticCpuKernelMod, int64_t);
MS_REG_CPU_KERNEL_T(
  AssignAdd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  SquaredDifference,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  ArithmeticCpuKernelMod, int);
MS_REG_CPU_KERNEL_T(
  SquaredDifference,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  ArithmeticCpuKernelMod, float16);
MS_REG_CPU_KERNEL_T(
  SquaredDifference,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  SquaredDifference,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
MS_REG_CPU_KERNEL_T(
  Atan2,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  ArithmeticCpuKernelMod, float);
MS_REG_CPU_KERNEL_T(
  Atan2,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  ArithmeticCpuKernelMod, double);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_CPU_KERNEL_H_
