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

#include "plugin/device/gpu/kernel/nn/adaptive_avg_pool2d_gpu_kernel.h"
#include "ops/adaptive_avg_pool_2d.h"

namespace mindspore {
namespace kernel {
constexpr uint kNumberTwo = 2;
constexpr uint kNumberThree = 2;

using KernelRunFunc = AdaptiveAvgPool2DKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &AdaptiveAvgPool2DKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &AdaptiveAvgPool2DKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &AdaptiveAvgPool2DKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &AdaptiveAvgPool2DKernelMod::LaunchKernel<double>}};
  return func_list;
}

template <typename T>
bool AdaptiveAvgPool2DKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                              const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  auto status = ApplyAdaptiveAvgPool2D(size_, input_height_, input_width_, output_height_, output_width_, input_addr,
                                       output_addr, reinterpret_cast<cudaStream_t>(stream_ptr_));

  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

bool AdaptiveAvgPool2DKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

bool AdaptiveAvgPool2DKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  size_t input_num = inputs.size();
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_num;
  }
  return true;
}

int AdaptiveAvgPool2DKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs[0]->GetShapeVector();
  auto output_shape = outputs[0]->GetShapeVector();
  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }
  len_ = static_cast<uint>(input_shape.size());
  if (len_ < kNumberTwo) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than " << kNumberTwo
                      << ", but got " << len_;
  }
  input_height_ = static_cast<uint>(input_shape[len_ - kNumberTwo]);
  input_width_ = static_cast<uint>(input_shape[len_ - 1]);
  size_ = static_cast<uint>(len_ == kNumberThree ? input_shape[0] : input_shape[0] * input_shape[1]);

  uint out_len = static_cast<uint>(output_shape.size());
  if (out_len < kNumberTwo) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than " << kNumberTwo
                      << ", but got " << out_len;
  }
  output_height_ = static_cast<uint>(output_shape[out_len - kNumberTwo]);
  output_width_ = static_cast<uint>(output_shape[out_len - 1]);
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdaptiveAvgPool2D, AdaptiveAvgPool2DKernelMod);
}  // namespace kernel
}  // namespace mindspore
