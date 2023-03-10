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

#include "plugin/device/gpu/kernel/nn/adagrad_gpu_kernel.h"
#include "mindspore/core/ops/apply_adagrad.h"
#include "mindspore/core/abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
bool AdagradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::ApplyAdagrad>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr_);
  kernel_name_ = kernel_ptr_->name();
  constexpr size_t input_num = 4;
  constexpr size_t output_num = 2;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int AdagradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto variable_shape = inputs[kIndex0]->GetShapeVector();
  auto accumulation_shape = inputs[kIndex1]->GetShapeVector();
  auto gradient_shape = inputs[kIndex3]->GetShapeVector();

  variable_shape_ = SizeOf(variable_shape);
  accumulation_shape_ = SizeOf(accumulation_shape);
  gradient_shape_ = SizeOf(gradient_shape);

  return KRET_OK;
}
template <typename T, typename S, typename G>
bool AdagradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  variable_size_ = variable_shape_ * sizeof(T);
  accumulation_size_ = accumulation_shape_ * sizeof(T);
  learning_rate_size_ = sizeof(S);
  gradient_size_ = gradient_shape_ * sizeof(G);
  T *variable = GetDeviceAddress<T>(inputs, kIndex0);
  T *accumulation = GetDeviceAddress<T>(inputs, kIndex1);
  S *learning_rate = GetDeviceAddress<S>(inputs, kIndex2);
  G *gradient = GetDeviceAddress<G>(inputs, kIndex3);
  T *variable_out = GetDeviceAddress<T>(outputs, kIndex0);
  T *accumulation_out = GetDeviceAddress<T>(outputs, kIndex1);

  auto status = ApplyAdagrad(inputs[0]->size / sizeof(T), update_slots, learning_rate, gradient, variable, accumulation,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(variable_out, variable, variable_size_, cudaMemcpyDeviceToDevice,
                                                     reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "cudaMemcpyAsync output failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(accumulation_out, accumulation, accumulation_size_, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync output failed");
  return true;
}

std::vector<std::pair<KernelAttr, AdagradGpuKernelMod::AdagradLaunchFunc>> AdagradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdagradGpuKernelMod::LaunchKernel<float, float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdagradGpuKernelMod::LaunchKernel<half, half, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdagradGpuKernelMod::LaunchKernel<float, half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdagradGpuKernelMod::LaunchKernel<half, float, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &AdagradGpuKernelMod::LaunchKernel<half, double, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &AdagradGpuKernelMod::LaunchKernel<float, double, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdagradGpuKernelMod::LaunchKernel<double, double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdagradGpuKernelMod::LaunchKernel<double, half, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &AdagradGpuKernelMod::LaunchKernel<double, float, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   &AdagradGpuKernelMod::LaunchKernel<utils::Complex<float>, utils::Complex<float>, utils::Complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   &AdagradGpuKernelMod::LaunchKernel<utils::Complex<double>, utils::Complex<double>, utils::Complex<double>>},
};

std::vector<KernelAttr> AdagradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, AdagradGpuKernelMod::AdagradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyAdagrad, AdagradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
