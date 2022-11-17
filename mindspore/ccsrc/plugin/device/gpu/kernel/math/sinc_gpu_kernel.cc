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

#include "plugin/device/gpu/kernel/math/sinc_gpu_kernel.h"
#include "mindspore/core/ops/sinc.h"
#include <functional>
#include <utility>
#include <map>
#include <string>
#include <complex>
#include <vector>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool SincGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::Sinc>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!kernel_ptr_) {
    MS_LOG(ERROR) << "cast Sinc ops failed!";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  return true;
}

int SincGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &others) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  size_t input_size = output_elements_ * unit_input_size_;
  size_t output_size = output_elements_ * unit_output_size_;
  input_size_list_.emplace_back(input_size);
  output_size_list_.emplace_back(output_size);
  return KRET_OK;
}

template <typename T, typename S>
bool SincGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  S *output = GetDeviceAddress<S>(outputs, 0);
  CalSinc(output_elements_, input, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, SincGpuKernelMod::SincFunc>> SincGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<uint8_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<int8_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<uint16_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<uint32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<uint64_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<bool, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SincGpuKernelMod::LaunchKernel<half, half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SincGpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &SincGpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &SincGpuKernelMod::LaunchKernel<Complex<float>, Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &SincGpuKernelMod::LaunchKernel<Complex<double>, Complex<double>>}};

std::vector<KernelAttr> SincGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SincFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Sinc, SincGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
