/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_LOGIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_LOGIC_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <limits>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ArithmeticLogicCPUKernel : public CPUKernel {
 public:
  ArithmeticLogicCPUKernel() = default;
  ~ArithmeticLogicCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitComputeFunc();
  void Less(const T *input1, const T *input2, bool *out) const;
  void Equal(const T *input1, const T *input2, bool *out) const;
  void NotEqual(const T *input1, const T *input2, bool *out) const;
  void Greater(const T *input1, const T *input2, bool *out) const;
  void GreaterEqual(const T *input1, const T *input2, bool *out) const;
  void LessEqual(const T *input1, const T *input2, bool *out) const;
  void LogicalAnd(const T *input1, const T *input2, bool *out) const;
  void LogicalOr(const T *input1, const T *input2, bool *out) const;

  using TypeComputeFunc = std::function<void(ArithmeticLogicCPUKernel *, const T *, const T *, bool *)>;
  TypeComputeFunc compute_func_{nullptr};
  size_t output_size_{1};
  TypeId dtype_{kTypeUnknown};

  std::vector<size_t> input_shape1_;
  std::vector<size_t> input_shape2_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> input_element_num2_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_element_num_;
};

MS_REG_CPU_KERNEL_T(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int);
MS_REG_CPU_KERNEL_T(
  Less, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  Less, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, bool);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int8_t);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int16_t);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, uint8_t);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, uint16_t);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, uint32_t);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float16);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  Equal, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, double);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, bool);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int8_t);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int16_t);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, uint8_t);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, uint16_t);
MS_REG_CPU_KERNEL_T(
  NotEqual, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, uint32_t);
MS_REG_CPU_KERNEL_T(
  NotEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float16);
MS_REG_CPU_KERNEL_T(
  NotEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  NotEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, double);
MS_REG_CPU_KERNEL_T(
  Greater, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int);
MS_REG_CPU_KERNEL_T(
  Greater,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  Greater, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int);
MS_REG_CPU_KERNEL_T(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  GreaterEqual,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(
  LessEqual, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int);
MS_REG_CPU_KERNEL_T(
  LessEqual, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(
  LessEqual,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  LogicalAnd, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, bool);
MS_REG_CPU_KERNEL_T(
  LogicalOr, KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
  ArithmeticLogicCPUKernel, bool);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ARITHMETIC_LOGIC_CPU_KERNEL_H_
