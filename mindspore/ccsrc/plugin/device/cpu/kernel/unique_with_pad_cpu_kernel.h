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
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniqueWithPadInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniqueWithPadOutputsNum, kernel_name_);
    int ret = UniqueCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
    if (ret != 0) {
      return ret;
    }
    is_need_retrieve_output_shape_ = false;
    if (batch_rank_ > 0) {
      auto pad_shape = inputs[kPadNumIndex]->GetShapeVector();
      auto pad_nums = std::accumulate(pad_shape.begin(), pad_shape.end(), 1, std::multiplies<int64_t>());
      if (pad_nums != static_cast<int64_t>(batch_size_)) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << "', the elements num of input 'pad' must be equal to input 'x' batch size, "
                             "but got the elements num of input 'pad': "
                          << Vector2Str(pad_shape) << " and input 'x' batch size: " << batch_size_;
      }
    }
    return ret;
  }

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
                 const std::vector<size_t> &start) {
    if (inputs.size() < kUniqueWithPadInputsNum || outputs.size() < kUniqueWithPadOutputsNum) {
      return;
    }
    auto pad_num_p = reinterpret_cast<T *>(inputs[1]->addr);
    auto *out = reinterpret_cast<T *>(outputs[0]->addr);
    for (size_t batch_i = 0; batch_i < batch_size_; batch_i++) {
      T pad_num = *pad_num_p;
      for (size_t i = start[batch_i]; i < input_size_; ++i) {
        out[i] = pad_num;
      }
      pad_num_p++;
      out += input_size_;
    }
  }
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UNIQUE_WITH_PAD_CPU_KERNEL_H_
