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

#include "kernel/gpu/cuda/nn/hshrink_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/hshrink_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHShrinkInputsNum = 2;
constexpr size_t kHShrinkOutputsNum = 1;
}  // namespace

bool HShrinkGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kHShrinkInputsNum || outputs.size() != kHShrinkOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kHShrinkInputsNum << " and "
                  << kHShrinkOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  return true;
}

int HShrinkGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.size() != kHShrinkInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 1.";
    return KRET_RESIZE_FAILED;
  }
  input_elements_ = inputs[0]->size() / unit_size_;
  lambd = inputs[kIndex1]->GetValueWithCheck<float>();
  return KRET_OK;
}

template <typename T>
bool HShrinkGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto status =
    CalHShrink(input_elements_, input, lambd, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, HShrinkGpuKernelMod::HShrinkFunc>> HShrinkGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16),
   &HShrinkGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &HShrinkGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> HShrinkGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, HShrinkFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, HShrink, HShrinkGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
