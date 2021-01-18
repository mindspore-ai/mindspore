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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
#include <memory>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class EltWiseGradCPUKernel : public CPUKernel {
 public:
  EltWiseGradCPUKernel() = default;
  ~EltWiseGradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  template <typename T>
  void ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end);
  template <typename T>
  void GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end);
  std::vector<size_t> input_shape0_;
  std::vector<size_t> input_shape1_;
  std::vector<size_t> input_element_num0_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_element_num_;
  OperateType operate_type_{RELUGRAD};
  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel);
MS_REG_CPU_KERNEL(
  ReLU6Grad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel);
MS_REG_CPU_KERNEL(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel);
MS_REG_CPU_KERNEL(
  SigmoidGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel);
MS_REG_CPU_KERNEL(
  SqrtGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel);
MS_REG_CPU_KERNEL(
  TanhGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel);
MS_REG_CPU_KERNEL(GeluGrad,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  EltWiseGradCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
