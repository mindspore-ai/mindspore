/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
enum ReductionType { kNone, kMean, kSum };
class BCEWithLogitsLossCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  BCEWithLogitsLossCpuKernelMod() = default;
  ~BCEWithLogitsLossCpuKernelMod() override;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  using BceFunc = std::function<bool(BCEWithLogitsLossCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  size_t input_size_{1};
  TypeId input_data_type_{kTypeUnknown};
  std::vector<size_t> input_logits_shape_;
  std::vector<size_t> input_label_shape_;
  std::vector<size_t> input_weight_shape_;
  std::vector<size_t> input_post_weight_shape_;
  ReductionType reduction_{kNone};
  BceFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, BceFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H
