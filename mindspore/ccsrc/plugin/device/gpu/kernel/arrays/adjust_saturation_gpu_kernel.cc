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

#include <algorithm>
#include <map>
#include <utility>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/arrays/adjust_saturation_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adjustsaturation_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_BUM = 2;
constexpr size_t OUTPUT_NUM = 1;

bool AdjustSaturationGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), INPUT_BUM, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), OUTPUT_NUM, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int AdjustSaturationGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  return KernelMod::Resize(base_operator, inputs, outputs);
}

template <typename T>
bool AdjustSaturationGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_image = GetDeviceAddress<T>(inputs, 0);
  float *saturation_scale = GetDeviceAddress<float>(inputs, 1);
  T *output_image = GetDeviceAddress<T>(outputs, 0);
  int input_element = inputs[0]->size / sizeof(T);
  auto status = CalAdjustSaturation(input_element, input_image, output_image, saturation_scale, device_id_,
                                    reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, AdjustSaturationGpuKernelMod::AdjustSaturationFunc>>
  AdjustSaturationGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64),
     &AdjustSaturationGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &AdjustSaturationGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat16),
     &AdjustSaturationGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> AdjustSaturationGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdjustSaturationFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdjustSaturation, AdjustSaturationGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
