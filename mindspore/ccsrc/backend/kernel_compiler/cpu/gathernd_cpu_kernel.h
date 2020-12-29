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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHERND_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHERND_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class GatherNdCPUKernel : public CPUKernel {
 public:
  GatherNdCPUKernel() = default;
  ~GatherNdCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  std::vector<size_t> input_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> output_shapes_;

  std::vector<size_t> dims_;
  std::vector<int> batch_indices_;
  std::vector<int> batch_strides_;

  TypeId dtype_{kTypeUnknown};
};

MS_REG_CPU_KERNEL(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  GatherNdCPUKernel);
MS_REG_CPU_KERNEL(
  GatherNd, KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  GatherNdCPUKernel);
MS_REG_CPU_KERNEL(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  GatherNdCPUKernel);
MS_REG_CPU_KERNEL(
  GatherNd,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  GatherNdCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_GATHERND_CPU_KERNEL_H_
