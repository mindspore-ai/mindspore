/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_WITH_PAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_WITH_PAD_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/unique_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class UniqueWithPadCpuKernelMod : public UniqueCpuKernelMod {
 public:
  UniqueWithPadCpuKernelMod() = default;
  ~UniqueWithPadCpuKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddOutputAttr(kNumberTypeInt32)
                                                     .AddOutputAttr(kNumberTypeInt32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeInt64),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeInt32)};
    return support_list;
  }

 private:
  inline static constexpr size_t kUniqueWithPadInputsNum = 2;
  inline static constexpr size_t kUniqueWithPadOutputsNum = 2;

  template <typename T>
  static void PadOutput(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs, size_t start) {
    if (inputs.size() < kUniqueWithPadInputsNum || outputs.size() < kUniqueWithPadOutputsNum) {
      return;
    }
    auto pad_num = *reinterpret_cast<T *>(inputs[1]->addr);
    auto *out = reinterpret_cast<T *>(outputs[0]->addr);
    size_t length = outputs[0]->size / sizeof(T);
    for (size_t i = start; i < length; ++i) {
      out[i] = pad_num;
    }
  }
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_WITH_PAD_CPU_KERNEL_H_
