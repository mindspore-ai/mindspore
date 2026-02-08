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

#include "kernel/gpu/cuda/nn/apply_add_sign_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/apply_add_sign_impl.cuh"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/curand.h"
#include "mindspore/ops/infer/apply_add_sign.h"

namespace mindspore {
namespace kernel {
bool ApplyAddSignGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  t_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  s_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);
  g_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex6).dtype);
  return true;
}

int ApplyAddSignGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> variable_shape_ = std::vector<int64_t>(inputs[kIndex0]->GetDeviceShapeVector().begin(),
                                                              inputs[kIndex0]->GetDeviceShapeVector().end());
  std::vector<int64_t> learning_rate_shape_ = std::vector<int64_t>(inputs[kIndex2]->GetDeviceShapeVector().begin(),
                                                                   inputs[kIndex2]->GetDeviceShapeVector().end());
  std::vector<int64_t> gradient_shape_ = std::vector<int64_t>(inputs[kIndex6]->GetDeviceShapeVector().begin(),
                                                              inputs[kIndex6]->GetDeviceShapeVector().end());
  t_elements_ = std::accumulate(variable_shape_.begin(), variable_shape_.end(), 1, std::multiplies<size_t>());
  s_elements_ = std::accumulate(learning_rate_shape_.begin(), learning_rate_shape_.end(), 1, std::multiplies<size_t>());
  g_elements_ = std::accumulate(gradient_shape_.begin(), gradient_shape_.end(), 1, std::multiplies<size_t>());
  is_null_input_ = (t_elements_ == 0 || s_elements_ == 0 || g_elements_ == 0);
  if (is_null_input_) {
    return 0;
  }
  size_t variable_size_ = t_elements_ * t_size_;
  size_t accumulation_size_ = t_elements_ * t_size_;
  output_size_list_.emplace_back(variable_size_);
  output_size_list_.emplace_back(accumulation_size_);
  return KRET_OK;
}

void ApplyAddSignGpuKernelMod::ResetResource() noexcept {
  t_elements_ = 0;
  s_elements_ = 0;
  g_elements_ = 0;
  is_null_input_ = false;
  output_size_list_.clear();
}

template <typename T, typename S, typename G>
bool ApplyAddSignGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs) {
  T *variable = GetDeviceAddress<T>(inputs, 0);
  T *accumulation = GetDeviceAddress<T>(inputs, 1);
  S *learning_rate = GetDeviceAddress<S>(inputs, 2);
  S *alpha = GetDeviceAddress<S>(inputs, 3);
  S *sign_decay = GetDeviceAddress<S>(inputs, 4);
  S *beta = GetDeviceAddress<S>(inputs, 5);
  G *gradient = GetDeviceAddress<G>(inputs, 6);
  T *variable_out = GetDeviceAddress<T>(outputs, 0);
  T *accumulation_out = GetDeviceAddress<T>(outputs, 1);
  auto status = ApplyAddSign(t_elements_, variable, accumulation, learning_rate, alpha, sign_decay, beta, gradient,
                             device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(variable_out, variable, outputs[kIndex0]->size(), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync output failed");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(accumulation_out, accumulation, outputs[kIndex1]->size(), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync output failed");
  return true;
}

std::vector<std::pair<KernelAttr, ApplyAddSignGpuKernelMod::ApplyAddSignFunc>> ApplyAddSignGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &ApplyAddSignGpuKernelMod::LaunchKernel<double, double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &ApplyAddSignGpuKernelMod::LaunchKernel<float, float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &ApplyAddSignGpuKernelMod::LaunchKernel<half, half, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &ApplyAddSignGpuKernelMod::LaunchKernel<half, float, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &ApplyAddSignGpuKernelMod::LaunchKernel<float, float, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &ApplyAddSignGpuKernelMod::LaunchKernel<float, half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &ApplyAddSignGpuKernelMod::LaunchKernel<float, half, half>}};

std::vector<KernelAttr> ApplyAddSignGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAddSignFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAddSign, ApplyAddSignGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
