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

#include "kernel/gpu/cuda/nn/maxpool_grad_grad_with_argmax_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/maxpool_grad_grad_with_argmax_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolGradGradWithArgmaxInputsNum = 3;
constexpr size_t kMaxPoolGradGradWithArgmaxOutputsNum = 1;
constexpr size_t kArgmaxIndex = 2;
}  // namespace

bool MaxPoolGradGradWithArgmaxGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kMaxPoolGradGradWithArgmaxInputsNum || outputs.size() != kMaxPoolGradGradWithArgmaxOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kMaxPoolGradGradWithArgmaxInputsNum
                  << " and " << kMaxPoolGradGradWithArgmaxOutputsNum << ", but get " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int MaxPoolGradGradWithArgmaxGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  auto in_shapes = inputs[0]->GetShapeVector();
  auto out_shapes = outputs[0]->GetShapeVector();
  batch_ = out_shapes[0];
  output_elements_ = std::accumulate(out_shapes.begin(), out_shapes.end(), 1, std::multiplies<size_t>());
  input_batch_stride_ = std::accumulate(in_shapes.begin() + 1, in_shapes.end(), 1, std::multiplies<size_t>());
  output_batch_stride_ = std::accumulate(out_shapes.begin() + 1, out_shapes.end(), 1, std::multiplies<size_t>());
  return KRET_OK;
}

template <typename T, typename I>
bool MaxPoolGradGradWithArgmaxGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &,
                                                         const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolGradGradWithArgmaxInputsNum, kernel_name_);

  T *grad_addr = GetDeviceAddress<T>(inputs, kIndex1);
  I *argmax_addr = GetDeviceAddress<I>(inputs, kArgmaxIndex);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);

  CalMaxPoolGradGradWithArgmax<T, I>(grad_addr, argmax_addr, batch_, input_batch_stride_, output_batch_stride_,
                                     output_addr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

const std::vector<std::pair<KernelAttr, MaxPoolGradGradWithArgmaxGpuKernelMod::KernelRunFunc>>
  &MaxPoolGradGradWithArgmaxGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, MaxPoolGradGradWithArgmaxGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradGradWithArgmaxGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradGradWithArgmaxGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolGradGradWithArgmaxGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolGradGradWithArgmaxGpuKernelMod::LaunchKernel<half, int64_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
