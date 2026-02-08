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

#include "kernel/gpu/cuda/nn/prelu_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool PReLUGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
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

int PReLUGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto weight_shape = LongVecToSizeVec(inputs[kIndex1]->GetShapeVector());

  auto weight_rank = weight_shape.size();
  if (weight_rank != 1) {
    MS_EXCEPTION(ValueError) << "The dimension of 'weight' must be 1";
  }
  auto x_rank = input_shape.size();
  auto channel_num = x_rank <= 1 ? 1 : input_shape[1];
  auto weight_len = weight_shape[0];
  if (weight_len != 1 && weight_len != channel_num) {
    MS_EXCEPTION(ValueError) << "For 'Prelu', the length of 'weight' must be equal to number of channels: "
                             << channel_num << ", but got " << weight_shape << ".";
  }

  input_length_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<>());
  per_channel_length_ =
    input_shape.size() <= 1 ? input_length_ : input_length_ / (input_shape[kIndex0] * input_shape[kIndex1]);
  weight_length_ = weight_shape[0];
  return KRET_OK;
}

template <typename T>
bool PReLUGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto input = GetDeviceAddress<T>(inputs, 0);
  auto weight = GetDeviceAddress<T>(inputs, 1);
  auto output = GetDeviceAddress<T>(outputs, 0);

  auto status = CalPReLU(input_length_, weight_length_, per_channel_length_, input, weight, output,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
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
