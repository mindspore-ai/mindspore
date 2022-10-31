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

#include "plugin/device/cpu/kernel/prelu_cpu_kernel.h"
#include <utility>
#include <algorithm>
#include "mindspore/core/ops/prelu.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool PReluCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::PReLU>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "cast PRelu ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int PReluCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, others); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto weight_shape = LongVecToSizeVec(inputs[kIndex1]->GetShapeVector());
  input_length_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<>());
  per_channel_length_ =
    input_shape.size() <= 1 ? input_length_ : input_length_ / (input_shape[kIndex0] * input_shape[kIndex1]);
  weight_length_ = weight_shape[0];
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, PReluCpuKernelMod::PReLULaunchFunc>> PReluCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &PReluCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &PReluCpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> PReluCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, PReluCpuKernelMod::PReLULaunchFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T>
bool PReluCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto *weight = reinterpret_cast<T *>(inputs[1]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(weight, false);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  auto task = [this, input, weight, output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t channel_id = weight_length_ == 1 ? 0 : (i / per_channel_length_) % weight_length_;
      output[i] = input[i] < static_cast<T>(0) ? weight[channel_id] * input[i] : input[i];
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_, pool_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PReLU, PReluCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
