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

#include "kernel/gpu/cudnn/convolution/conv2d_forward_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/pad_impl.cuh"

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

bool Conv2dFwdGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

bool Conv2dFwdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return InitialAttributes(primitive_, &conv_args_, inputs);
}

int Conv2dFwdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[0]->GetDeviceShapeVector();
  auto filter_shape = inputs[kIndex1]->GetDeviceShapeVector();
  auto output_shape = outputs[0]->GetDeviceShapeVector();
  if (!CheckTensorSize({input_shape, filter_shape, output_shape})) {
    return KRET_RESIZE_FAILED;
  }
  if (!conv_kernel_ptr) {
    SetConvolutionInChannel(&conv_args_, input_shape);
    auto conv_kernel_type = SelectConvolutionGpuKernel(conv_args_);
    conv_kernel_ptr =
      ConvolutionGpuKernelFactory::CreateConvolutionGpuKernel(conv_args_, conv_kernel_type, ConvType::kForward);

    InitResource();
  }
  ResetResource();
  // for dynamic pad in dynamic infer shape
  conv_args_.pad_list.clear();
  auto pad_list_attr = GetValue<std::vector<int64_t>>(primitive_->GetAttr("pad_list"));
  if (pad_list_attr.size() != kConv2dInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got "
                      << pad_list_attr.size();
  }
  std::transform(pad_list_attr.begin(), pad_list_attr.end(), std::back_inserter(conv_args_.pad_list),
                 [](const int64_t &value) { return static_cast<int>(value); });
  return conv_kernel_ptr->InitialKernel(&conv_args_, input_shape, filter_shape, output_shape, &input_size_list_,
                                        &output_size_list_, &workspace_size_list_);
}

template <typename T>
bool Conv2dFwdGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *filter_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  return conv_kernel_ptr->LaunchKernel<T>(conv_args_, input_addr, filter_addr, output_addr, workspace, stream_ptr_);
}

void Conv2dFwdGpuKernelMod::InitResource() {
  if (conv_kernel_ptr) {
    conv_kernel_ptr->InitResource();
  }
}

void Conv2dFwdGpuKernelMod::DestroyResource() noexcept {
  if (conv_kernel_ptr) {
    conv_kernel_ptr->DestroyResource();
  }
}

void Conv2dFwdGpuKernelMod::ResetResource() noexcept {
  if (conv_kernel_ptr) {
    conv_kernel_ptr->ResetResource(&conv_args_, &input_size_list_, &output_size_list_, &workspace_size_list_);
  }
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Conv2D, Conv2dFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
