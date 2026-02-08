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
#include "kernel/gpu/cudnn/convolution/conv2d_input_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/pad_impl.cuh"

namespace mindspore {
namespace kernel {
using KernelRunFunc = Conv2dInputGradGpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &Conv2dInputGradGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &Conv2dInputGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &Conv2dInputGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &Conv2dInputGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &Conv2dInputGradGpuKernelMod::LaunchKernel<half>}};
  return func_list;
}

bool Conv2dInputGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

bool Conv2dInputGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  auto prim = primitive_;
  MS_EXCEPTION_IF_NULL(prim);
  const auto &inplace_algo_attr = prim->GetAttr("inplace_algo");
  auto inplace_algo_value = inplace_algo_attr == nullptr ? "cover" : GetValue<std::string>(inplace_algo_attr);
  conv_args_.beta = inplace_algo_value == "cover" ? 0 : 1;

  return InitialAttributes(primitive_, &conv_args_, inputs);
}

int Conv2dInputGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto dy_shape = inputs[0]->GetDeviceShapeVector();
  auto filter_shape = inputs[kIndex1]->GetDeviceShapeVector();
  std::vector<int64_t> input_shape;
  input_shape = inputs[kShapeIndex]->GetValueWithCheck<std::vector<int64_t>>();

  if (!CheckTensorSize({input_shape, dy_shape, filter_shape})) {
    return KRET_RESIZE_FAILED;
  }

  if (!conv_kernel_ptr) {
    if (conv_args_.data_format_attr == kOpFormat_NCHW && conv_args_.data_format == kOpFormat_NHWC) {
      ShapeNCHW2NHWC(&input_shape);
    }

    SetConvolutionInChannel(&conv_args_, input_shape);
    auto conv_kernel_type = SelectConvolutionGpuKernel(conv_args_);
    conv_kernel_ptr =
      ConvolutionGpuKernelFactory::CreateConvolutionGpuKernel(conv_args_, conv_kernel_type, ConvType::kInputGrad);

    InitResource();
  }
  ResetResource();
  return conv_kernel_ptr->InitialKernel(&conv_args_, dy_shape, input_shape, filter_shape, &input_size_list_,
                                        &output_size_list_, &workspace_size_list_);
}

template <typename T>
bool Conv2dInputGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs) {
  T *dy = GetDeviceAddress<T>(inputs, 0);
  T *w = GetDeviceAddress<T>(inputs, 1);
  T *dx = GetDeviceAddress<T>(outputs, 0);

  return conv_kernel_ptr->LaunchKernel<T>(conv_args_, dy, w, dx, workspace, stream_ptr_);
}

void Conv2dInputGradGpuKernelMod::InitResource() {
  if (conv_kernel_ptr) {
    conv_kernel_ptr->InitResource();
  }
}

void Conv2dInputGradGpuKernelMod::DestroyResource() noexcept {
  if (conv_kernel_ptr) {
    conv_kernel_ptr->DestroyResource();
  }
}

void Conv2dInputGradGpuKernelMod::ResetResource() noexcept {
  if (conv_kernel_ptr) {
    conv_kernel_ptr->ResetResource(&conv_args_, &input_size_list_, &output_size_list_, &workspace_size_list_);
  }
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Conv2DBackpropInput, Conv2dInputGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
