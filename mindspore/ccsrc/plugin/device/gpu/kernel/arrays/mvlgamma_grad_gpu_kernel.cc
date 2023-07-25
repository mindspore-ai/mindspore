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

#include "plugin/device/gpu/kernel/arrays/mvlgamma_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool MvlgammaGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::MvlgammaGrad>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
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
  p_ = kernel_ptr_->get_p();
  return true;
}

int MvlgammaGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just
    // return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  if (input_elements_ == 0) {
    is_null_input_ = true;
  }
  int64_t input_dims = input_shape.size();
  if (input_dims < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'x' should be at least 1-D, but got "
                      << input_dims << "-D.";
    return KRET_RESIZE_FAILED;
  }
  size_t input_size = input_elements_ * unit_size_;

  input_size_list_.push_back(input_size);
  input_size_list_.push_back(input_size);
  output_size_list_.push_back(input_size);
  return KRET_OK;
}

template <typename T>
bool MvlgammaGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  T *y_grad = GetDeviceAddress<T>(inputs, 0);
  T *x = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  auto status =
    CalMvlgammaGrad(input_elements_, y_grad, x, p_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, MvlgammaGradGpuKernelMod::MvlgammaGradFunc>> MvlgammaGradGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &MvlgammaGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &MvlgammaGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MvlgammaGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MvlgammaGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MvlgammaGrad, MvlgammaGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
