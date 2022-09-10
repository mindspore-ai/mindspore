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

#include "plugin/device/gpu/kernel/math/polar_gpu_kernel.h"
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kTwo = 2;
}  // namespace

bool PolarGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::Polar>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', valid gpu kernel does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  return true;
}

int PolarGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  for (const auto &output : outputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto output_shape = output->GetShapeVector();
    if (!IsValidShape(output_shape)) {
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
  input_size_list_.push_back(output_elements_ * unit_input_size_);
  input_size_list_.push_back(output_elements_ * unit_input_size_);
  output_size_list_.push_back(output_elements_ * unit_output_size_);

  return KRET_OK;
}

template <typename T, typename S>
bool PolarGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  T *abs = GetDeviceAddress<T>(inputs, 0);
  T *angle = GetDeviceAddress<T>(inputs, 1);
  S *output = GetDeviceAddress<S>(outputs, 0);
  CalPolar(output_elements_, abs, angle, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return True;
}

template <typename R>
using Complex = mindspore::utils::Complex<R>;

std::vector<std::pair<KernelAttr, PolarGpuKernelMod::Polarfunc>> PolarGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
   &PolarGpuKernelMod::LaunchKernel<float, Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
   &PolarGpuKernelMod::LaunchKernel<double, Complex<double>>}};

std::vector<KernelAttr> PolarGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, Polarfunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Polar, PolarGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
