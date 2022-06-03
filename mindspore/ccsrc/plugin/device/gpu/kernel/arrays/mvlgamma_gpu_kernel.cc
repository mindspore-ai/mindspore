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

#include "plugin/device/gpu/kernel/arrays/mvlgamma_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool MvlgammaGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::Mvlgamma>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  p_ = kernel_ptr_->get_p();
  if (p_ < 1) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the attr 'p' has to be greater than or equal to 1, "
                  << "but got " << p_ << ".";
    return false;
  }
  return true;
}

int MvlgammaGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                          inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  if (input_elements_ == 0) {
    is_null_input_ = true;
  }
  int64_t input_dims = input_shape.size();
  if (input_dims < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'x' should be at least 1-D, but got " << input_dims
                  << "-D.";
    return KRET_RESIZE_FAILED;
  }
  size_t input_size = input_elements_ * unit_size_;
  input_size_list_.push_back(input_size);
  output_size_list_.push_back(input_size);
  workspace_size_list_.push_back(sizeof(bool));
  return KRET_OK;
}

template <typename T>
bool MvlgammaGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  bool *valid_d = GetDeviceAddress<bool>(workspace, 0);
  bool valid = true;
  bool *valid_h = &valid;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(valid_d, valid_h, sizeof(bool), cudaMemcpyHostToDevice,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cudaMemcpyAsync valid Host to Device failed.");
  CalMvlgamma(valid_d, input_elements_, input, p_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(valid_h, valid_d, sizeof(bool), cudaMemcpyDeviceToHost,
                                                     reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "cudaMemcpyAsync valid Device to Host failed.");
  if (!*valid_h) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", all element must be greater than (p-1)/2.";
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, MvlgammaGpuKernelMod::MvlgammaFunc>> MvlgammaGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &MvlgammaGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &MvlgammaGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MvlgammaGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MvlgammaFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Mvlgamma, MvlgammaGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
