/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/convolution/conv2d_forward_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"

namespace mindspore {
namespace kernel {
using KernelRunFunc = Conv2dFwdGpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &Conv2dFwdGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &Conv2dFwdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &Conv2dFwdGpuKernelMod::LaunchKernel<half>}};
  return func_list;
}

bool Conv2dFwdGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

bool Conv2dFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  return InitialAttributes(&conv_args_, base_operator, inputs);
}

int Conv2dFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[0]->GetDeviceShapeAdaptively();
  auto filter_shape = inputs[kIndex1]->GetDeviceShapeAdaptively();
  auto output_shape = outputs[0]->GetDeviceShapeAdaptively();
  if (!CheckTensorSize({input_shape, filter_shape, output_shape})) {
    return KRET_RESIZE_FAILED;
  }
  if (!conv_kernel_ptr_) {
    SetConvolutionInChannel(&conv_args_, input_shape);
    auto conv_kernel_type = SelectConvolutionGpuKernel(conv_args_);
    conv_kernel_ptr_ =
      ConvolutionGpuKernelFactory::CreateConvolutionGpuKernel(conv_args_, conv_kernel_type, ConvType::kForward);
    MS_EXCEPTION_IF_NULL(conv_kernel_ptr_);
    InitResource();
  }
  ResetResource();
  // for dynamic pad in dynamic infer shape
  conv_args_.pad_list.clear();
  auto pad_list_attr = GetValue<std::vector<int64_t>>(base_operator->GetAttr("pad_list"));
  if (pad_list_attr.size() != kConv2dInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got "
                      << pad_list_attr.size();
  }
  std::transform(pad_list_attr.begin(), pad_list_attr.end(), std::back_inserter(conv_args_.pad_list),
                 [](const int64_t &value) { return static_cast<int>(value); });
  return conv_kernel_ptr_->InitialKernel(&conv_args_, input_shape, filter_shape, output_shape, &input_size_list_,
                                         &output_size_list_, &workspace_size_list_);
}

template <typename T>
bool Conv2dFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *filter_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  MS_EXCEPTION_IF_NULL(conv_kernel_ptr_);
  return conv_kernel_ptr_->LaunchKernel<T>(conv_args_, input_addr, filter_addr, output_addr, workspace, stream_ptr_);
}

void Conv2dFwdGpuKernelMod::InitResource() {
  if (conv_kernel_ptr_) {
    conv_kernel_ptr_->InitResource();
  }
}

void Conv2dFwdGpuKernelMod::DestroyResource() noexcept {
  if (conv_kernel_ptr_) {
    conv_kernel_ptr_->DestroyResource();
  }
}

void Conv2dFwdGpuKernelMod::ResetResource() noexcept {
  if (conv_kernel_ptr_) {
    conv_kernel_ptr_->ResetResource(&conv_args_, &input_size_list_, &output_size_list_, &workspace_size_list_);
  }
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Conv2D, Conv2dFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
