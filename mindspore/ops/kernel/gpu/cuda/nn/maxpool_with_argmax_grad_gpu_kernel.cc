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

#include "kernel/gpu/cuda/nn/maxpool_with_argmax_grad_gpu_kernel.h"
#include <functional>
#include <memory>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolGradWithArgmaxInputsNum = 3;
constexpr size_t kMaxPoolGradWithArgmaxOutputsNum = 1;
constexpr size_t kXDimLowerLimit = 4;
constexpr size_t kDyDimLowerLimit = 4;
constexpr size_t kXIndexForN = 0;
constexpr size_t kXIndexForC = 1;
constexpr size_t kXIndexForH = 2;
constexpr size_t kXIndexForW = 3;
constexpr size_t kDyIndexForH = 2;
constexpr size_t kDyIndexForW = 3;
constexpr size_t Index2 = 2;
}  // namespace

template <typename T, typename S>
bool MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &workspace,
                                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *dy_addr = GetDeviceAddress<T>(inputs, 1);
  S *index_addr = GetDeviceAddress<S>(inputs, Index2);
  T *dx_addr = GetDeviceAddress<T>(outputs, 0);
  auto status = CalMaxPoolWithArgmaxGrad(dy_addr, index_addr, n_, c_, x_height_, x_width_, dy_height_, dy_width_,
                                         dx_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool MaxPoolGradWithArgmaxGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
}

bool MaxPoolGradWithArgmaxGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kMaxPoolGradWithArgmaxInputsNum || outputs.size() != kMaxPoolGradWithArgmaxOutputsNum) {
    MS_LOG(EXCEPTION) << kernel_name_ << ": input and output size must be " << kMaxPoolGradWithArgmaxInputsNum
                      << " and " << kMaxPoolGradWithArgmaxOutputsNum << ", but get " << inputs.size() << " and "
                      << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[pair.second].second;
  return true;
}

int MaxPoolGradWithArgmaxGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  // modified
  int ret = 0;
  if (ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> x_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> dy_shape = inputs[1]->GetShapeVector();
  std::vector<int64_t> index_shape = inputs[Index2]->GetShapeVector();
  std::vector<int64_t> dx_shape = outputs[0]->GetShapeVector();

  is_null_input_ = CHECK_SHAPE_NULL(x_shape, kernel_name_, "x") || CHECK_SHAPE_NULL(dy_shape, kernel_name_, "dy") ||
                   CHECK_SHAPE_NULL(index_shape, kernel_name_, "index") ||
                   CHECK_SHAPE_NULL(dx_shape, kernel_name_, "dx");
  if (is_null_input_) {
    return true;
  }
  if (x_shape.size() < kXDimLowerLimit || dy_shape.size() < kDyDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of x and dy cannot be less than 4, but got "
                      << "the dimension of x: " << x_shape.size() << ", the dimension of dy: " << dy_shape.size();
  }
  n_ = LongToSizeClipNeg(x_shape[kXIndexForN]);
  c_ = LongToSizeClipNeg(x_shape[kXIndexForC]);
  x_height_ = LongToSizeClipNeg(x_shape[kXIndexForH]);
  x_width_ = LongToSizeClipNeg(x_shape[kXIndexForW]);
  dy_height_ = LongToSizeClipNeg(dy_shape[kDyIndexForH]);
  dy_width_ = LongToSizeClipNeg(dy_shape[kDyIndexForW]);
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, MaxPoolGradWithArgmaxGpuKernelMod::MaxPoolGradWithArgmaxFunc>>
  MaxPoolGradWithArgmaxGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<uint64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &MaxPoolGradWithArgmaxGpuKernelMod::LaunchKernel<double, int32_t>},
};
std::vector<KernelAttr> MaxPoolGradWithArgmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolGradWithArgmaxFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPoolGradWithArgmax, MaxPoolGradWithArgmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
