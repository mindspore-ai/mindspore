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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class EmbeddingLookUpCPUKernel : public CPUKernel {
 public:
  EmbeddingLookUpCPUKernel() {}
  ~EmbeddingLookUpCPUKernel() override {}

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  template <typename T>
  void LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

 protected:
  void CheckParam(const CNodePtr &kernel_node);
  int64_t offset_{0};
  size_t indices_lens_{1};
  size_t first_dim_size_{1};
  size_t outer_dim_size_{1};
  TypeId indices_data_type_{kNumberTypeInt32};
  CNodeWeakPtr node_wpt_;
};

MS_REG_CPU_KERNEL(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  EmbeddingLookUpCPUKernel);

MS_REG_CPU_KERNEL(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  EmbeddingLookUpCPUKernel);

MS_REG_CPU_KERNEL(
  EmbeddingLookup,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  EmbeddingLookUpCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EMBEDDING_LOOK_UP_CPU_KERNEL_H_
