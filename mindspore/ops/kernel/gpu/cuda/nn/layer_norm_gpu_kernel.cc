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

#include "kernel/gpu/cuda/nn/layer_norm_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include "kernel/gpu/cuda_impl/cuda_ops/layer_norm_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLayerNormInputXIndex = 0;
constexpr size_t kLayerNormInputGammaIndex = 1;
constexpr size_t kLayerNormInputBetaIndex = 2;
constexpr size_t kLayerNormInputBeginNormAxisIndex = 3;
constexpr size_t kLayerNormInputBeginParamsAxisIndex = 4;
constexpr size_t kLayerNormInputEpsilonIndex = 5;
constexpr size_t kLayerNormOutputYIndex = 0;
constexpr size_t kLayerNormOutputMeanIndex = 1;
constexpr size_t kLayerNormOutputVarIndex = 2;
}  // namespace
bool LayerNormGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  epsilon_ = inputs[kLayerNormInputEpsilonIndex]->GetValueWithCheck<float_t>();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LayerNormGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid LayerNormGpuKernelMod input size!";
  }
  auto begin_norm_axis = inputs[kLayerNormInputBeginNormAxisIndex]->GetValueWithCheck<int64_t>();
  auto begin_params_axis = inputs[kLayerNormInputBeginParamsAxisIndex]->GetValueWithCheck<int64_t>();
  auto input_shape = inputs[kLayerNormInputXIndex]->GetShapeVector();
  if (begin_norm_axis < 0) {
    begin_norm_axis += input_shape.size();
  }

  if (begin_params_axis < 0) {
    begin_params_axis += input_shape.size();
  }

  if (LongToSize(begin_norm_axis) > input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'begin_norm_axis' must be less than or equal "
                      << "to the dimension of input_x, but got begin_norm_axis: " << LongToSize(begin_norm_axis)
                      << ", the dimension of input_x: " << input_shape.size();
  }
  input_row_ =
    std::accumulate(input_shape.begin(), input_shape.begin() + LongToSize(begin_norm_axis), 1, std::multiplies<int>());
  input_col_ =
    std::accumulate(input_shape.begin() + LongToSize(begin_norm_axis), input_shape.end(), 1, std::multiplies<int>());
  param_dim_ =
    std::accumulate(input_shape.begin() + LongToSize(begin_params_axis), input_shape.end(), 1, std::multiplies<int>());
  return ret;
}

bool LayerNormGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  kernel_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void LayerNormGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  auto x = GetDeviceAddress<T>(inputs, kLayerNormInputXIndex);
  auto gamma = GetDeviceAddress<T>(inputs, kLayerNormInputGammaIndex);
  auto beta = GetDeviceAddress<T>(inputs, kLayerNormInputBetaIndex);
  auto y = GetDeviceAddress<T>(outputs, kLayerNormOutputYIndex);
  auto mean = GetDeviceAddress<float>(outputs, kLayerNormOutputMeanIndex);
  auto variance = GetDeviceAddress<float>(outputs, kLayerNormOutputVarIndex);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(gamma);
  MS_EXCEPTION_IF_NULL(beta);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(variance);

  auto status =
    LayerNorm(input_row_, input_col_, param_dim_, epsilon_, x, gamma, beta, y, mean, variance, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
}

std::vector<std::pair<KernelAttr, LayerNormGpuKernelMod::KernelFunc>> LayerNormGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LayerNormGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LayerNorm, LayerNormGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
