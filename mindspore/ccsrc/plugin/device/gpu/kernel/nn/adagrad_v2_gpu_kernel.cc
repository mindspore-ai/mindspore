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
#include <mindspore/core/abstract/utils.h>
#include <memory>
#include <utility>
#include <algorithm>
#include "abstract/utils.h"
#include "mindspore/core/ops/apply_adagrad_v2.h"
#include "plugin/device/gpu/kernel/nn/adagrad_v2_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adagrad_v2_impl.cuh"

namespace mindspore {
namespace kernel {
void AdagradV2GpuKernelMod::InOutputResize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  t_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  s_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);

  std::vector<int64_t> variable_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> accumulation_shape_ = std::vector<int64_t>(
    inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(), inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> gradient_shape_ = std::vector<int64_t>(inputs.at(kIndex3)->GetDeviceShapeAdaptively().begin(),
                                                              inputs.at(kIndex3)->GetDeviceShapeAdaptively().end());
  input_elements_ = std::accumulate(variable_shape_.begin(), variable_shape_.end(), 1, std::multiplies<int64_t>());

  is_null_input_ = (input_elements_ == 0);

  if (is_null_input_) {
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    input_size_list_.push_back(0);
    output_size_list_.push_back(0);
    output_size_list_.push_back(0);
    return;
  }

  variable_size_ = t_size_;
  accumulation_size_ = t_size_;
  learning_rate_size_ = s_size_;
  gradient_size_ = t_size_;

  for (int64_t i = 0; i < static_cast<int64_t>(variable_shape_.size()); i++) {
    variable_size_ *= variable_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(accumulation_shape_.size()); i++) {
    accumulation_size_ *= accumulation_shape_[i];
  }
  for (int64_t i = 0; i < static_cast<int64_t>(gradient_shape_.size()); i++) {
    gradient_size_ *= gradient_shape_[i];
  }
  input_size_list_.push_back(variable_size_);
  input_size_list_.push_back(accumulation_size_);
  input_size_list_.push_back(learning_rate_size_);
  input_size_list_.push_back(gradient_size_);
  output_size_list_.push_back(variable_size_);
  output_size_list_.push_back(accumulation_size_);
}

bool AdagradV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::ApplyAdagradV2>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  epsilon_ = kernel_ptr_->get_epsilon();
  update_slots_ = kernel_ptr_->get_update_slots();
  constexpr int INPUT_NUM = 4;
  if (inputs.size() != INPUT_NUM) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 4, but got " << inputs.size();
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

int AdagradV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  kernel_ptr_ = base_operator;
  InOutputResize(base_operator, inputs, outputs);
  outputs_ = outputs;
  return KRET_OK;
}

template <typename T, typename S>
bool AdagradV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *variable = GetDeviceAddress<T>(inputs, kIndex0);
  T *accumulation = GetDeviceAddress<T>(inputs, kIndex1);
  S *learning_rate = GetDeviceAddress<S>(inputs, kIndex2);
  T *gradient = GetDeviceAddress<T>(inputs, kIndex3);
  T *variable_out = GetDeviceAddress<T>(outputs, kIndex0);
  T *accumulation_out = GetDeviceAddress<T>(outputs, kIndex1);
  auto status = ApplyAdagradV2(size_t(inputs[0]->size / sizeof(T)), epsilon_, update_slots_, variable, accumulation,
                               learning_rate, gradient, device_id_, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(variable_out, variable, variable_size_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "cudaMemcpyAsync output failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(accumulation_out, accumulation, accumulation_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "cudaMemcpyAsync output failed");
  return true;
}

std::vector<KernelAttr> AdagradV2GpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;

  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ApplyAdagradV2Func> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, AdagradV2GpuKernelMod::ApplyAdagradV2Func>> AdagradV2GpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdagradV2GpuKernelMod::LaunchKernel<double, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdagradV2GpuKernelMod::LaunchKernel<float, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdagradV2GpuKernelMod::LaunchKernel<half, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdagradV2GpuKernelMod::LaunchKernel<half, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdagradV2GpuKernelMod::LaunchKernel<float, half>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdagradV2GpuKernelMod::LaunchKernel<float, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdagradV2GpuKernelMod::LaunchKernel<half, double>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdagradV2GpuKernelMod::LaunchKernel<double, float>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdagradV2GpuKernelMod::LaunchKernel<double, half>}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdagradV2, AdagradV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
