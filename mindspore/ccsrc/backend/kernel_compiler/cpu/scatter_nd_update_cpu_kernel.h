/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
struct ComputeParams {
  T *x_{nullptr};
  int *indices_{nullptr};
  T *updates_{nullptr};
  int unit_size_{0};
  int indices_unit_rank_{0};
  std::vector<int> *out_strides_{nullptr};
  size_t x_mem_size_{0};
};

class ScatterNdUpdateCPUKernel : public CPUKernel {
 public:
  ScatterNdUpdateCPUKernel() = default;
  ~ScatterNdUpdateCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  TypeId dtype_{kTypeUnknown};
  int unit_size_{0};
  size_t num_units_{0};
  int indices_unit_rank_{0};
  std::vector<int> out_strides_;
};

MS_REG_CPU_KERNEL(ScatterNdUpdate,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  ScatterNdUpdateCPUKernel);

MS_REG_CPU_KERNEL(TensorScatterUpdate,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  ScatterNdUpdateCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ND_UPDATE_CPU_KERNEL_H_
