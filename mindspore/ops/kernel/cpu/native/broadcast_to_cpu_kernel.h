/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BROADCAST_TO_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BROADCAST_TO_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <string>

#include "kernel/cpu/cpu_kernel.h"
#include "primitive/array_ops.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/cpu/nnacl/base/broadcast_to.h"

namespace mindspore {
namespace kernel {
namespace broadcast_to_cpu {
constexpr auto kBroadcastTo = "BroadcastTo";
constexpr auto kDynamicBroadcastTo = "DynamicBroadcastTo";
constexpr auto kUnknown = "Unknown";
class BroadcastToCpuKernelMod : public NativeCpuKernelMod {
 public:
  BroadcastToCpuKernelMod() = default;
  explicit BroadcastToCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~BroadcastToCpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  void CheckArgs();

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using BroadcastToFunc =
    std::function<bool(BroadcastToCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::map<std::string, std::vector<std::pair<KernelAttr, BroadcastToFunc>>> func_list_;
  BroadcastToFunc kernel_func_;

  void InitTaskFunc(const CNodePtr &kernel_node);
  ShapeVector input_shape_;
  ShapeVector output_shape_;
  BroadcastShapeInfo shape_info_{};
  std::string kernel_type_{kUnknown};
};
}  // namespace broadcast_to_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BROADCAST_TO_CPU_KERNEL_H_
