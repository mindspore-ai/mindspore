/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESIZE_NEAREST_NEIGHBOR_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESIZE_NEAREST_NEIGHBOR_GRAD_CPU_KERNEL_H_

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ResizeNearestNeighborGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  ResizeNearestNeighborGradCpuKernelMod() = default;
  ~ResizeNearestNeighborGradCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat16)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat16),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat64),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeInt32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat16)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat16),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeFloat64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeFloat64),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeInt32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  template <typename T>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  TypeId dtype_{kTypeUnknown};
  bool align_corners_{false};
  size_t batch_size_{0};
  size_t channel_{0};
  size_t in_height_{0};
  size_t in_width_{0};
  size_t out_height_{0};
  size_t out_width_{0};
  float height_scale_{1.0};
  float width_scale_{1.0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RESIZE_NEAREST_NEIGHBOR_GRAD_CPU_KERNEL_H_
