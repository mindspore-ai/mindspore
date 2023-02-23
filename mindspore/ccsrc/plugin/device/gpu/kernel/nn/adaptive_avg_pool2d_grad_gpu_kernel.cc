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

#include "plugin/device/gpu/kernel/nn/adaptive_avg_pool2d_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool AdaptiveAvgPool2DGradKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdaptiveAvgPool2dGradInputNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int AdaptiveAvgPool2DGradKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto grad_shape = inputs[kIndex1]->GetShapeVector();     // dy
  auto output_shape = outputs[kIndex0]->GetShapeVector();  // dx

  auto input_rank = grad_shape.size();
  auto output_rank = output_shape.size();
  if (input_rank < kAdaptiveAvgPool2dGradMinRank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input cannot be less than "
                  << kAdaptiveAvgPool2dGradMinRank << ", but got " << input_rank;
    return KRET_RESIZE_FAILED;
  }
  if (output_rank < kAdaptiveAvgPool2dGradMinRank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of output cannot be less than "
                  << kAdaptiveAvgPool2dGradMinRank << ", but got " << output_rank;
    return KRET_RESIZE_FAILED;
  }
  input_height_ = static_cast<uint>(grad_shape[input_rank - kDim2]);
  input_width_ = static_cast<uint>(grad_shape[input_rank - 1]);
  size_ = static_cast<uint>(input_rank == (kAdaptiveAvgPool2dGradMinRank + 1) ? grad_shape[0]
                                                                              : grad_shape[0] * grad_shape[1]);

  output_height_ = static_cast<uint>(output_shape[output_rank - kDim2]);
  output_width_ = static_cast<uint>(output_shape[output_rank - 1]);

  workspace_size_ = sizeof(float);  // used as float buffer when T is half
  workspace_size_ *= SizeOf(output_shape);
  workspace_size_list_.push_back(workspace_size_);
  return KRET_OK;
}

template <typename T>
bool AdaptiveAvgPool2DGradKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &workspace,
                                                  const std::vector<AddressPtr> &outputs) {
  T *dy_addr = GetDeviceAddress<T>(inputs, 1);
  T *dx_addr = GetDeviceAddress<T>(outputs, 0);
  float *wk_addr = GetDeviceAddress<float>(workspace, 0);

  auto status = ApplyAdaptiveAvgPool2DGrad(size_, input_height_, input_width_, output_height_, output_width_, dy_addr,
                                           dx_addr, wk_addr, reinterpret_cast<cudaStream_t>(stream_ptr_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

const std::vector<std::pair<KernelAttr, AdaptiveAvgPool2DGradKernelMod::KernelRunFunc>>
  &AdaptiveAvgPool2DGradKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, AdaptiveAvgPool2DGradKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &AdaptiveAvgPool2DGradKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &AdaptiveAvgPool2DGradKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &AdaptiveAvgPool2DGradKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdaptiveAvgPool2DGrad, AdaptiveAvgPool2DGradKernelMod);
}  // namespace kernel
}  // namespace mindspore
