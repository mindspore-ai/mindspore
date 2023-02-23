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

#include "plugin/device/gpu/kernel/nn/adadelta_gpu_kernel.h"

namespace mindspore {
namespace kernel {
void AdadeltaGpuKernelMod::InOutputResize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  input_size_list_.clear();
  output_size_list_.clear();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  t_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  s_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex3).dtype);
  g_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex6).dtype);

  std::vector<int64_t> variable_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> accumulation_shape_ = std::vector<int64_t>(
    inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(), inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> accumulation_update_shape_ = std::vector<int64_t>(
    inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(), inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> gradient_shape_ = std::vector<int64_t>(inputs.at(kIndex6)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex6)->GetDeviceShapeAdaptively().end());
  input_elements_ = std::accumulate(variable_shape_.begin(), variable_shape_.end(), 1, std::multiplies<int64_t>());

  is_null_input_ = (input_elements_ == 0);

  if (is_null_input_) {
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
    return;
  }

  variable_size_ = t_size_;
  accumulation_size_ = t_size_;
  accumulation_update_size_ = t_size_;
  learning_rate_size_ = s_size_;
  rho_size_ = s_size_;
  epsilon_size_ = s_size_;
  gradient_size_ = g_size_;

  for (int64_t i = 0; i < static_cast<int64_t>(variable_shape_.size()); i++) {
    variable_size_ *= variable_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(accumulation_shape_.size()); i++) {
    accumulation_size_ *= accumulation_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(accumulation_update_shape_.size()); i++) {
    accumulation_update_size_ *= accumulation_update_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(gradient_shape_.size()); i++) {
    gradient_size_ *= gradient_shape_[i];
  }

  input_size_list_.push_back(variable_size_);
  input_size_list_.push_back(accumulation_size_);
  input_size_list_.push_back(accumulation_update_size_);
  input_size_list_.push_back(learning_rate_size_);
  input_size_list_.push_back(rho_size_);
  input_size_list_.push_back(epsilon_size_);
  input_size_list_.push_back(gradient_size_);
  output_size_list_.push_back(variable_size_);
  output_size_list_.push_back(accumulation_size_);
  output_size_list_.push_back(accumulation_update_size_);
}

bool AdadeltaGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  kernel_ptr_ = std::make_shared<ops::ApplyAdadelta>(base_operator->GetPrim());
  constexpr int INPUT_NUM = 7;
  if (inputs.size() != INPUT_NUM) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 7, but got " << inputs.size();
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' dose not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  InOutputResize(base_operator, inputs, outputs);
  outputs_ = outputs;
  return true;
}

int AdadeltaGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  kernel_ptr_ = base_operator;
  InOutputResize(base_operator, inputs, outputs);
  outputs_ = outputs;
  return KRET_OK;
}

template <typename T, typename S, typename G>
bool AdadeltaGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  T *variable = GetDeviceAddress<T>(inputs, kIndex0);
  T *accumulation = GetDeviceAddress<T>(inputs, kIndex1);
  T *accumulation_update = GetDeviceAddress<T>(inputs, kIndex2);
  S *learning_rate = GetDeviceAddress<S>(inputs, kIndex3);
  S *rho = GetDeviceAddress<S>(inputs, kIndex4);
  S *epsilon = GetDeviceAddress<S>(inputs, kIndex5);
  G *gradient = GetDeviceAddress<G>(inputs, kIndex6);
  T *variable_out = GetDeviceAddress<T>(outputs, kIndex0);
  T *accumulation_out = GetDeviceAddress<T>(outputs, kIndex1);
  T *accumulation_update_out = GetDeviceAddress<T>(outputs, kIndex2);

  auto status =
    ApplyAdadelta(inputs[0]->size / sizeof(T), learning_rate, rho, epsilon, gradient, variable, accumulation,
                  accumulation_update, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(variable_out, variable, variable_size_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaMemcpyAsync output failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(accumulation_out, accumulation, accumulation_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync output failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(accumulation_update_out, accumulation_update, accumulation_update_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync output failed");
  return true;
}

std::vector<KernelAttr> AdadeltaGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAdadeltaFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, AdadeltaGpuKernelMod::ApplyAdadeltaFunc>> AdadeltaGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdadeltaGpuKernelMod::LaunchKernel<double, double, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdadeltaGpuKernelMod::LaunchKernel<float, float, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdadeltaGpuKernelMod::LaunchKernel<half, half, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdadeltaGpuKernelMod::LaunchKernel<half, float, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdadeltaGpuKernelMod::LaunchKernel<float, float, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdadeltaGpuKernelMod::LaunchKernel<float, half, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdadeltaGpuKernelMod::LaunchKernel<half, half, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdadeltaGpuKernelMod::LaunchKernel<float, half, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdadeltaGpuKernelMod::LaunchKernel<double, half, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdadeltaGpuKernelMod::LaunchKernel<double, float, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdadeltaGpuKernelMod::LaunchKernel<double, float, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdadeltaGpuKernelMod::LaunchKernel<double, half, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdadeltaGpuKernelMod::LaunchKernel<half, float, float>}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdadelta, AdadeltaGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
