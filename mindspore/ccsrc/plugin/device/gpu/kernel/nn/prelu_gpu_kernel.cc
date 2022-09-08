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

#include "plugin/device/gpu/kernel/nn/prelu_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool PReLUGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
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

int PReLUGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
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

template <typename T>
bool PReLUGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto input = GetDeviceAddress<T>(inputs, 0);
  auto weight = GetDeviceAddress<T>(inputs, 1);
  auto output = GetDeviceAddress<T>(outputs, 0);

  CalPReLU(input_length_, weight_length_, per_channel_length_, input, weight, output,
           reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, PReLUGpuKernelMod::PReLULaunchFunc>> PReLUGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &PReLUGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &PReLUGpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> PReLUGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, PReLUGpuKernelMod::PReLULaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, PReLU, PReLUGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
