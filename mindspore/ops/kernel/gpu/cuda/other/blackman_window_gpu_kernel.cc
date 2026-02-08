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

#include "kernel/gpu/cuda/other/blackman_window_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool BlackmanWindowGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [int32, int64], "
                  << "but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  periodic_ = GetValue<bool>(primitive_->GetAttr(ops::kPeriodic));
  return true;
}

int BlackmanWindowGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                          inputs.at(kIndex0)->GetDeviceShapeVector().end());
  int64_t input_dims = input_shape.size();
  if (input_dims != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' must be 0-D, but got " << input_dims << "-D.";
    return false;
  }
  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeVector().end());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  size_t output_size = output_elements_ * unit_output_size_;
  output_size_list_.push_back(output_size);
  return KRET_OK;
}

template <typename T, typename S>
bool BlackmanWindowGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  S *output = GetDeviceAddress<S>(outputs, 0);
  CalBlackmanWindow(output_elements_, input, periodic_, output, device_id_,
                    reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, BlackmanWindowGpuKernelMod::BmWFunc>> BlackmanWindowGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &BlackmanWindowGpuKernelMod::LaunchKernel<int, half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &BlackmanWindowGpuKernelMod::LaunchKernel<int64_t, half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &BlackmanWindowGpuKernelMod::LaunchKernel<int, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &BlackmanWindowGpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &BlackmanWindowGpuKernelMod::LaunchKernel<int, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &BlackmanWindowGpuKernelMod::LaunchKernel<int64_t, double>}};

std::vector<KernelAttr> BlackmanWindowGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BmWFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BlackmanWindow, BlackmanWindowGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
