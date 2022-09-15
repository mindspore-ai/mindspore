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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RANDOM_CHOICE_WITH_MASK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RANDOM_CHOICE_WITH_MASK_CPU_KERNEL_H_
#include <vector>
#include <random>
#include <algorithm>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class RandomChoiceWithMaskCpuKernelMod : public NativeCpuKernelMod {
 public:
  RandomChoiceWithMaskCpuKernelMod() = default;
  ~RandomChoiceWithMaskCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool)};
    return support_list;
  }

 private:
  int32_t input_dim_size_ = 0;
  int32_t input_total_count_ = 1;
  int32_t count_{0};
  std::vector<int32_t> dims_;
  size_t input_shape_size_{0};
  size_t seed_{0};
  size_t seed2_{0};
  std::mt19937 generator_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RANDOM_CHOICE_WITH_MASK_CPU_KERNEL_H_
