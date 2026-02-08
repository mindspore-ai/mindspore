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

#include "kernel/gpu/cuda/nn/layer_norm_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include "kernel/gpu/cuda_impl/cuda_ops/layer_norm_grad_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLayerNormGradInputXIndex = 0;
constexpr size_t kLayerNormGradInputDyIndex = 1;
constexpr size_t kLayerNormGradInputVarIndex = 2;
constexpr size_t kLayerNormGradInputMeanIndex = 3;
constexpr size_t kLayerNormGradInputGammaIndex = 4;
constexpr size_t kLayerNormGradBeginNormAxisIndex = 5;
constexpr size_t kLayerNormGradBeginParamsAxisIndex = 6;
constexpr size_t kLayerNormGradOutputDxIndex = 0;
constexpr size_t kLayerNormGradOutputDgIndex = 1;
constexpr size_t kLayerNormGradOutputDbIndex = 2;
}  // namespace

bool LayerNormGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For GPU '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  if (!primitive()->HasAttr(ops::kEpsilon)) {
    MS_LOG(WARNING) << "LayerNormGrad should have attr 'epsilon'.";
  } else {
    epsilon_ = GetValue<float>(primitive()->GetAttr(ops::kEpsilon));
  }

  return true;
}

int LayerNormGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid LayerNormGradGpuKernelMod input size!";
  }
  auto begin_norm_axis = inputs[kLayerNormGradBeginNormAxisIndex]->GetValueWithCheck<int64_t>();
  auto begin_params_axis = inputs[kLayerNormGradBeginParamsAxisIndex]->GetValueWithCheck<int64_t>();
  auto input_shape = inputs[kLayerNormGradInputXIndex]->GetShapeVector();
  if (begin_norm_axis < 0) {
    begin_norm_axis += input_shape.size();
  }

  if (begin_params_axis < 0) {
    begin_params_axis += input_shape.size();
  }

  if (LongToSize(begin_norm_axis) > input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'begin_norm_axis' must be less than or equal "
                      << "to the dimension of input, but got begin_norm_axis: " << LongToSize(begin_norm_axis)
                      << ", the dimension of input: " << input_shape.size();
  }

  input_row_ =
    std::accumulate(input_shape.begin(), input_shape.begin() + LongToSize(begin_norm_axis), 1, std::multiplies<int>());
  input_col_ =
    std::accumulate(input_shape.begin() + LongToSize(begin_norm_axis), input_shape.end(), 1, std::multiplies<int>());
  param_dim_ =
    std::accumulate(input_shape.begin() + LongToSize(begin_params_axis), input_shape.end(), 1, std::multiplies<int>());

  return ret;
}

bool LayerNormGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  kernel_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void LayerNormGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto x = GetDeviceAddress<T>(inputs, kLayerNormGradInputXIndex);
  auto dy = GetDeviceAddress<T>(inputs, kLayerNormGradInputDyIndex);
  auto var = GetDeviceAddress<float>(inputs, kLayerNormGradInputVarIndex);
  auto mean = GetDeviceAddress<float>(inputs, kLayerNormGradInputMeanIndex);
  auto gamma = GetDeviceAddress<T>(inputs, kLayerNormGradInputGammaIndex);
  auto dx = GetDeviceAddress<T>(outputs, kLayerNormGradOutputDxIndex);
  auto dg = GetDeviceAddress<T>(outputs, kLayerNormGradOutputDgIndex);
  auto db = GetDeviceAddress<T>(outputs, kLayerNormGradOutputDbIndex);

  auto status =
    LayerNormGrad(input_row_, input_col_, param_dim_, epsilon_, dy, x, mean, var, gamma, dx, dg, db, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
}

std::vector<std::pair<KernelAttr, LayerNormGradGpuKernelMod::KernelFunc>> LayerNormGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &LayerNormGradGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &LayerNormGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LayerNormGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LayerNormGrad, LayerNormGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
