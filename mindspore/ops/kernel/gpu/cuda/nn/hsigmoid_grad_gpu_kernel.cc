/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/nn/hsigmoid_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "mindspore/ops/infer/ops_func_impl/hsigmoid_grad.h"

namespace mindspore {
namespace kernel {
constexpr auto kHSigmoidGrad = "HSigmoidGrad";
constexpr const size_t kHSigmoidGradInputsNum = 2;
constexpr const size_t kHSigmoidGradOutputsNum = 1;

template <typename T>
bool HSigmoidGradGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                            const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHSigmoidGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHSigmoidGradOutputsNum, kernel_name_);
  T *dy_addr = GetDeviceAddress<T>(inputs, 0);
  T *x_addr = GetDeviceAddress<T>(inputs, 1);
  T *dx_addr = GetDeviceAddress<T>(outputs, 0);

  auto status = CalHSigmoidGrad(static_cast<size_t>(input_elements_), dy_addr, x_addr, dx_addr,
                                reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, HSigmoidGradGpuKernelMod::HSigmoidGradLaunchFunc>>
  HSigmoidGradGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &HSigmoidGradGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &HSigmoidGradGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &HSigmoidGradGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &HSigmoidGradGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &HSigmoidGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &HSigmoidGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &HSigmoidGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> HSigmoidGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, HSigmoidGradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool HSigmoidGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kHSigmoidGradInputsNum || outputs.size() != kHSigmoidGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kHSigmoidGradInputsNum << " and "
                  << kHSigmoidGradOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[pair.second].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  return true;
}

int HSigmoidGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape_1 = inputs[0]->GetShapeVector();
  std::vector<int64_t> input_shape_2 = inputs[1]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  auto in_shape_size_1 = input_shape_1.size();
  if (in_shape_size_1 > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input should be less than or equal to max_dims 7, but got "
                      << in_shape_size_1 << ".";
    return KRET_RESIZE_FAILED;
  }
  auto in_shape_size_2 = input_shape_2.size();
  auto output_shape_size = output_shape.size();
  if (in_shape_size_1 != output_shape_size || in_shape_size_1 != in_shape_size_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input one shape size should be the same as input two shape size and"
                  << " output shape size, but got input one shape size " << in_shape_size_1 << " input two shape size "
                  << in_shape_size_2 << " output shape size" << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  // A Code Block For setting input and output shape.
  {
    input_shape_ = std::vector<size_t>(inputs[kIndex0]->GetDeviceShapeVector().begin(),
                                       inputs[kIndex0]->GetDeviceShapeVector().end());
    input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
    is_null_input_ = (input_elements_ == 0);
  }
  return KRET_OK;
}
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, HSigmoidGrad,
                                 []() { return std::make_shared<HSigmoidGradGpuKernelMod>(kHSigmoidGrad); });
}  // namespace kernel
}  // namespace mindspore
