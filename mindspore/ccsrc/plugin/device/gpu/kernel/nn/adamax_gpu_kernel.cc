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

#include "plugin/device/gpu/kernel/nn/adamax_gpu_kernel.h"

namespace mindspore {
namespace kernel {
void AdamaxGpuKernelMod::InOutputResize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  input_size_list_.clear();
  output_size_list_.clear();

  std::vector<int64_t> variable_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> m_shape_ = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                                       inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> v_shape_ = std::vector<int64_t>(inputs.at(kIndex2)->GetDeviceShapeAdaptively().begin(),
                                                       inputs.at(kIndex2)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> gradient_shape_ = std::vector<int64_t>(inputs.at(kIndex8)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex8)->GetDeviceShapeAdaptively().end());
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
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
    return;
  }

  variable_size_ = t_size_;
  m_size_ = t_size_;
  v_size_ = t_size_;

  beta1_power_size_ = s_size_;
  learning_rate_size_ = s_size_;
  beta1_size_ = s_size_;
  beta2_size_ = s_size_;
  epsilon_size_ = s_size_;

  gradient_size_ = g_size_;

  for (int64_t i = 0; i < static_cast<int64_t>(variable_shape_.size()); i++) {
    variable_size_ *= variable_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(m_shape_.size()); i++) {
    m_size_ *= m_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(v_shape_.size()); i++) {
    v_size_ *= v_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(gradient_shape_.size()); i++) {
    gradient_size_ *= gradient_shape_[i];
  }

  input_size_list_.push_back(variable_size_);
  input_size_list_.push_back(m_size_);
  input_size_list_.push_back(v_size_);
  input_size_list_.push_back(beta1_power_size_);
  input_size_list_.push_back(learning_rate_size_);
  input_size_list_.push_back(beta1_size_);
  input_size_list_.push_back(beta2_size_);
  input_size_list_.push_back(epsilon_size_);
  input_size_list_.push_back(gradient_size_);
  output_size_list_.push_back(variable_size_);
  output_size_list_.push_back(m_size_);
  output_size_list_.push_back(v_size_);
}

bool AdamaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  kernel_ptr_ = std::make_shared<ops::ApplyAdaMax>(base_operator->GetPrim());
  constexpr int INPUT_NUM = 9;
  if (inputs.size() != INPUT_NUM) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 9, but got " << inputs.size();
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' dose not support this kernel type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[index].second;
  t_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  s_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex3).dtype);
  g_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex8).dtype);

  InOutputResize(base_operator, inputs, outputs);
  outputs_ = outputs;
  return true;
}

int AdamaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  InOutputResize(base_operator, inputs, outputs);
  kernel_ptr_ = base_operator;
  outputs_ = outputs;
  return KRET_OK;
}

template <typename T, typename S, typename G>
bool AdamaxGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  T *variable = GetDeviceAddress<T>(inputs, kIndex0);
  T *m = GetDeviceAddress<T>(inputs, kIndex1);
  T *v = GetDeviceAddress<T>(inputs, kIndex2);
  S *beta1_power = GetDeviceAddress<S>(inputs, kIndex3);
  S *learning_rate = GetDeviceAddress<S>(inputs, kIndex4);
  S *beta1 = GetDeviceAddress<S>(inputs, kIndex5);
  S *beta2 = GetDeviceAddress<S>(inputs, kIndex6);
  S *epsilon = GetDeviceAddress<S>(inputs, kIndex7);
  G *gradient = GetDeviceAddress<G>(inputs, kIndex8);
  T *variable_out = GetDeviceAddress<T>(outputs, kIndex0);
  T *m_out = GetDeviceAddress<T>(outputs, kIndex1);
  T *v_out = GetDeviceAddress<T>(outputs, kIndex2);

  auto status = ApplyAdamax(inputs[0]->size / sizeof(T), beta1_power, learning_rate, beta1, beta2, epsilon, gradient,
                            variable, m, v, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(variable_out, variable, variable_size_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaMemcpyAsync output failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(m_out, m, m_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync output failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(v_out, v, v_size_, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync output failed");
  return true;
}

std::vector<KernelAttr> AdamaxGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAdamaxFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, AdamaxGpuKernelMod::ApplyAdamaxFunc>> AdamaxGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
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
   &AdamaxGpuKernelMod::LaunchKernel<double, double, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
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
   &AdamaxGpuKernelMod::LaunchKernel<float, float, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
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
   &AdamaxGpuKernelMod::LaunchKernel<half, half, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdamaxGpuKernelMod::LaunchKernel<half, float, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
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
   &AdamaxGpuKernelMod::LaunchKernel<float, float, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdamaxGpuKernelMod::LaunchKernel<float, half, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdamaxGpuKernelMod::LaunchKernel<float, half, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdamaxGpuKernelMod::LaunchKernel<double, half, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdamaxGpuKernelMod::LaunchKernel<double, float, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdamaxGpuKernelMod::LaunchKernel<double, float, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdamaxGpuKernelMod::LaunchKernel<double, half, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdamaxGpuKernelMod::LaunchKernel<half, float, float>}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdaMax, AdamaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
