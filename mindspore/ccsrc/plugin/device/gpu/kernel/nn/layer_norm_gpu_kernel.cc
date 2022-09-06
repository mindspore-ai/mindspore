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

#include "plugin/device/gpu/kernel/nn/layer_norm_gpu_kernel.h"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/layer_norm_impl.cuh"
#include "mindspore/core/ops/layer_norm.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLayerNormInputXIndex = 0;
constexpr size_t kLayerNormInputGammaIndex = 1;
constexpr size_t kLayerNormInputBetaIndex = 2;
constexpr size_t kLayerNormOutputYIndex = 0;
constexpr size_t kLayerNormOutputMeanIndex = 1;
constexpr size_t kLayerNormOutputVarIndex = 2;
}  // namespace
bool LayerNormGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::LayerNorm>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Cast ops::LayerNorm failed!";
  }
  kernel_name_ = kernel_ptr->name();
  epsilon_ = kernel_ptr->get_epsilon();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LayerNormGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::LayerNorm>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Cast ops::LayerNorm failed!";
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid LayerNormGpuKernelMod input size!";
  }
  auto begin_norm_axis = kernel_ptr->get_begin_norm_axis();
  auto begin_params_axis = kernel_ptr->get_begin_params_axis();
  auto input_shape = inputs[kLayerNormInputXIndex]->GetShapeVector();

  if (begin_norm_axis < 0) {
    begin_norm_axis += input_shape.size();
  }

  if (begin_params_axis < 0) {
    begin_params_axis += input_shape.size();
  }

  if (IntToSize(begin_norm_axis) > input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'begin_norm_axis' must be less than or equal "
                      << "to the dimension of input_x, but got begin_norm_axis: " << IntToSize(begin_norm_axis)
                      << ", the dimension of input_x: " << input_shape.size();
  }
  for (size_t i = 0; i < IntToSize(begin_norm_axis); i++) {
    input_row_ *= input_shape[i];
  }

  for (size_t i = begin_norm_axis; i < input_shape.size(); i++) {
    input_col_ *= input_shape[i];
  }

  for (size_t i = begin_params_axis; i < input_shape.size(); i++) {
    param_dim_ *= input_shape[i];
  }

  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  return ret;
}

bool LayerNormGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  kernel_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void LayerNormGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto x = GetDeviceAddress<T>(inputs, kLayerNormInputXIndex);
  auto gamma = GetDeviceAddress<T>(inputs, kLayerNormInputGammaIndex);
  auto beta = GetDeviceAddress<T>(inputs, kLayerNormInputBetaIndex);
  auto y = GetDeviceAddress<T>(outputs, kLayerNormOutputYIndex);
  auto mean = GetDeviceAddress<float>(outputs, kLayerNormOutputMeanIndex);
  auto variance = GetDeviceAddress<float>(outputs, kLayerNormOutputVarIndex);

  LayerNorm(input_row_, input_col_, param_dim_, epsilon_, x, gamma, beta, y, mean, variance, cuda_stream_);
}

std::vector<std::pair<KernelAttr, LayerNormGpuKernelMod::KernelFunc>> LayerNormGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> LayerNormGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LayerNorm, LayerNormGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
