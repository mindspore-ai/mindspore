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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSOR_ARRAY_STACK_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSOR_ARRAY_STACK_KERNEL_H_

#include <string>
#include <vector>
#include <atomic>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class TensorArrayCPUStackKernel : public CPUKernel {
 public:
  TensorArrayCPUStackKernel();
  ~TensorArrayCPUStackKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  void InitKernel(const CNodePtr &kernel_node) override;

 protected:
  void PostExecute();
  void ResetResource() noexcept;

 private:
  CNodeWeakPtr kernel_node_;
  int64_t handle_;
  size_t value_size_;
  size_t ele_size_;
  std::vector<size_t> shapes_;
  TypePtr type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
                  TensorArrayCPUStackKernel);
MS_REG_CPU_KERNEL(TensorArrayStack, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
                  TensorArrayCPUStackKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSOR_ARRAY_STACK_KERNEL_H_
