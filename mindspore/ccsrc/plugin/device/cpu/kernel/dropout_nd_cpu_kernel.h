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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DROPOUT_ND_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DROPOUT_ND_CPU_KERNEL_H_
#include <memory>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DropoutNdCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  DropoutNdCpuKernelMod() = default;
  ~DropoutNdCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  using DropoutNdFunc =
    std::function<bool(DropoutNdCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  void CheckDropOutNdShape();

  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> mask_shape_;
  size_t n_{1};
  size_t c_{1};
  size_t channels_{1};
  size_t element_nums_{1};
  size_t element_per_channel_{1};
  TypeId input_data_dtype_{kTypeUnknown};
  float keep_prob_{0.0};
  DropoutNdFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, DropoutNdFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DROPOUT_ND_CPU_KERNEL_H_
