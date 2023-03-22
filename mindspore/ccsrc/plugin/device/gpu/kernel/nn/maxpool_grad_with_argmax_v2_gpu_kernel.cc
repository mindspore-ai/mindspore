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

#include "plugin/device/gpu/kernel/nn/maxpool_grad_with_argmax_v2_gpu_kernel.h"
#include <functional>
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool_grad_with_argmax_v2_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kMaxPoolGradWithArgmaxV2 = "MaxPoolGradWithArgmaxV2";
constexpr size_t kInputShapeSize = 4;
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;

template <typename T, typename S>
bool MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *dy_addr = GetDeviceAddress<T>(inputs, kIndex1);
  S *index_addr = GetDeviceAddress<S>(inputs, kIndex2);
  T *dx_addr = GetDeviceAddress<T>(outputs, kIndex0);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(dx_addr, 0, outputs[kIndex0]->size, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'MaxPoolWithArgmaxGradV2' failed to cudaMemsetAsync");
  CalMaxPoolGradWithArgmaxV2(dy_addr, index_addr, x_hw_, dy_hw_, dy_nchw_, dx_addr, device_id_,
                             reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

bool MaxPoolGradWithArgmaxV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MaxPoolGradWithArgmaxV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (inputs.size() != kInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be " << kInputNum << ", but got "
                  << inputs.size();
    return KRET_RESIZE_FAILED;
  }
  if (outputs.size() != kOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs should be " << kOutputNum << ", but got "
                  << outputs.size();
    return KRET_RESIZE_FAILED;
  }
  auto x_shape = inputs.at(kIndex0)->GetShapeVector();
  auto dy_shape = inputs.at(kIndex1)->GetShapeVector();
  auto index_shape = inputs.at(kIndex2)->GetShapeVector();
  auto dx_shape = outputs.at(kIndex0)->GetShapeVector();

  is_null_input_ = CHECK_SHAPE_NULL(x_shape, kernel_name_, "x") || CHECK_SHAPE_NULL(dy_shape, kernel_name_, "dy") ||
                   CHECK_SHAPE_NULL(index_shape, kernel_name_, "index") ||
                   CHECK_SHAPE_NULL(dx_shape, kernel_name_, "dx");
  if (is_null_input_) {
    return KRET_RESIZE_FAILED;
  }
  if (x_shape.size() != kInputShapeSize || dy_shape.size() != kInputShapeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of x and dy should be equal to " << kInputShapeSize
                  << ", but got the dimension of x: " << x_shape.size() << ", the dimension of dy: " << dy_shape.size();
    return KRET_RESIZE_FAILED;
  }

  constexpr size_t nc_offset = 2;
  x_hw_ = std::accumulate(x_shape.begin() + nc_offset, x_shape.end(), 1, std::multiplies<int64_t>());
  dy_hw_ = std::accumulate(dy_shape.begin() + nc_offset, dy_shape.end(), 1, std::multiplies<int64_t>());
  dy_nchw_ = std::accumulate(dy_shape.begin(), dy_shape.end(), 1, std::multiplies<int64_t>());
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, MaxPoolGradWithArgmaxV2GpuKernelMod::MaxPoolArgmaxV2GradFunc>>
  MaxPoolGradWithArgmaxV2GpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint64_t, int32_t>},

    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &MaxPoolGradWithArgmaxV2GpuKernelMod::LaunchKernel<uint64_t, int64_t>},
};

std::vector<KernelAttr> MaxPoolGradWithArgmaxV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPoolArgmaxV2GradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MaxPoolGradWithArgmaxV2, []() {
  return std::make_shared<MaxPoolGradWithArgmaxV2GpuKernelMod>(kMaxPoolGradWithArgmaxV2);
});
}  // namespace kernel
}  // namespace mindspore
