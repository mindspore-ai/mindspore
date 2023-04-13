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
#include <map>
#include <functional>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/unique_cpu_kernel.h"

namespace mindspore {
namespace kernel {
inline static constexpr size_t kUniqueWithPadInputsNum = 2;
inline static constexpr size_t kUniqueWithPadOutputsNum = 2;
inline static constexpr size_t kPadNumIndex = 1;
inline static constexpr size_t kInputIndex = 0;
class UniqueWithPadCpuKernelMod : public UniqueCpuKernelMod {
 public:
  UniqueWithPadCpuKernelMod() = default;
  ~UniqueWithPadCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    dtype_ = inputs[0]->GetDtype();
    auto batch_rank = base_operator->get_batch_rank();
    if (batch_rank < 0) {
      return false;
    }
    batch_rank_ = static_cast<size_t>(batch_rank);
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

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
  template <typename T>
  void PadOutput(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                 const std::vector<size_t> &start);
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_WITH_PAD_CPU_KERNEL_H_
