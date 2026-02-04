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

#include "kernel/gpu/cuda/nn/adaptive_max_pool2d_gpu_kernel.h"
#include "mindspore/ops/infer/ops_func_impl/adaptive_max_pool2d.h"

namespace mindspore {
namespace kernel {

using KernelRunFunc = AdaptiveMaxPool2DKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &AdaptiveMaxPool2DKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeInt64),
     &AdaptiveMaxPool2DKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &AdaptiveMaxPool2DKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64),
     &AdaptiveMaxPool2DKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &AdaptiveMaxPool2DKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64),
     &AdaptiveMaxPool2DKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &AdaptiveMaxPool2DKernelMod::LaunchKernel<double>}};
  return func_list;
}

template <typename T>
bool AdaptiveMaxPool2DKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &,
                                              const std::vector<KernelTensor *> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  int64_t *indices_addr = nullptr;
  indices_addr = GetDeviceAddress<int64_t>(outputs, 1);

  auto status = ApplyAdaptiveMaxPool2D(size_, input_height_, input_width_, output_height_, output_width_, input_addr,
                                       output_addr, indices_addr, reinterpret_cast<cudaStream_t>(stream_ptr_));

  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool AdaptiveMaxPool2DKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

bool AdaptiveMaxPool2DKernelMod::InitSize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  if (inputs.size() != 2) {
    MS_LOG(EXCEPTION) << "For primitive[AdaptiveMaxPool2D], the size of input should be 2, but got " << inputs.size();
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);

  auto input_shape = inputs[0]->GetShapeVector();
  len_ = static_cast<size_t>(input_shape.size());
  if (len_ == 1 && input_shape[0] == ops::kDynamicRankValue) {
    return true;
  }

  if (len_ != ops::kFormatCHWShapeSize && len_ != ops::kFormatNCHWShapeSize) {
    MS_LOG(EXCEPTION) << "For primitive[AdaptiveMaxPool2D], the shape size of input argument[input_x] must "
                         "be 3 or 4, but got:"
                      << len_;
    return false;
  }

  auto output_size = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (output_size.size() == ops::kOutputSizeAttrSize) {
    // If the output size is none, the output shapes should be same as the input.
    output_height_ = (output_size[0] != ops::kPyValueNone ? static_cast<size_t>(output_size[0]) : input_height_);
    output_width_ = (output_size[1] != ops::kPyValueNone ? static_cast<size_t>(output_size[1]) : input_width_);
  } else {
    MS_LOG(EXCEPTION) << "For primitive[AdaptiveMaxPool2D], the size of attr[output_size] should be 2, but got:"
                      << output_size.size();
    return false;
  }

  // Check the parameters valid.

  input_height_ = static_cast<size_t>(input_shape[len_ - ops::kOutputSizeAttrSize]);
  input_width_ = static_cast<size_t>(input_shape[len_ - ops::kOutputSizeAttrSize + 1]);
  size_ = static_cast<size_t>(len_ == ops::kFormatCHWShapeSize ? input_shape[0] : input_shape[0] * input_shape[1]);
  input_size_ = sizeof(TypeIdToType(inputs[kIndex0]->dtype_id()));
  for (size_t i = 0; i < len_; i++) {
    input_size_ *= input_shape[i];
  }

  return true;
}

bool AdaptiveMaxPool2DKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return InitSize(inputs, outputs);
}

int AdaptiveMaxPool2DKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (!InitSize(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, AdaptiveMaxPool2D, AdaptiveMaxPool2DKernelMod);
}  // namespace kernel
}  // namespace mindspore
