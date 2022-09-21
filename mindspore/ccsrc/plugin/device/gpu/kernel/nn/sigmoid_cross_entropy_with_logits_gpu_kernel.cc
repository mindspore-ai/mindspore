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

#include "plugin/device/gpu/kernel/nn/sigmoid_cross_entropy_with_logits_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sigmoid_cross_entropy_with_logits_impl.cuh"

namespace mindspore {
namespace kernel {
bool SigmoidCrossEntropyWithLogitsGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T, typename S>
bool SigmoidCrossEntropyWithLogitsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<AddressPtr> &workspace,
                                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *logits_addr = GetDeviceAddress<T>(inputs, 0);
  S *labels_addr = GetDeviceAddress<S>(inputs, 1);
  T *outputs_addr = GetDeviceAddress<T>(outputs, 0);

  SigmoidCrossEntropyWithLogits(inputs[0]->size / sizeof(T), logits_addr, labels_addr, outputs_addr,
                                reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<KernelAttr> SigmoidCrossEntropyWithLogitsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SigmoidCrossEntropyWithLogitsLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, SigmoidCrossEntropyWithLogitsGpuKernelMod::SigmoidCrossEntropyWithLogitsLaunchFunc>>
  SigmoidCrossEntropyWithLogitsGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SigmoidCrossEntropyWithLogitsGpuKernelMod::LaunchKernel<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SigmoidCrossEntropyWithLogitsGpuKernelMod::LaunchKernel<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SigmoidCrossEntropyWithLogitsGpuKernelMod::LaunchKernel<half, half>},
};
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
