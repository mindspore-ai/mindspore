/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_TO_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_TO_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <string>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/base/broadcast_to.h"

namespace mindspore {
namespace kernel {
constexpr auto kBroadcastTo = "BroadcastTo";
constexpr auto kDynamicBroadcastTo = "DynamicBroadcastTo";
constexpr auto kUnknown = "Unknown";
class BroadcastToCpuKernelMod : public NativeCpuKernelMod {
 public:
  BroadcastToCpuKernelMod() = default;
  explicit BroadcastToCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~BroadcastToCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  void InitKernel(const CNodePtr &kernel_node) override;

  void CheckArgs();

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using BroadcastToFunc =
    std::function<bool(BroadcastToCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::map<std::string, std::vector<std::pair<KernelAttr, BroadcastToFunc>>> func_list_;
  BroadcastToFunc kernel_func_;

  void InitTaskFunc(const CNodePtr &kernel_node);
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  BroadcastShapeInfo shape_info_{};
  std::string kernel_type_{kUnknown};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_TO_CPU_KERNEL_H_
