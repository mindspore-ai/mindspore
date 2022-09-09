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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_V2_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod() = default;
  ~SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node);

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename I, typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using SparseSoftmaxCrossEntropyWithLogitsV2Func =
    std::function<bool(SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsV2Func>> func_list_;
  SparseSoftmaxCrossEntropyWithLogitsV2Func kernel_func_;

  std::vector<int64_t> features_shape;
  std::vector<int64_t> labels_shape;
  std::vector<int64_t> loss_shape;
  std::vector<int64_t> backprop_shape;
  std::string reduction;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_V2_CPU_KERNEL_H_
