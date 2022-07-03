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

#include "plugin/device/gpu/kernel/nn/maxpool3d_grad_with_argmax_gpu_kernel.h"
#include <functional>
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/maxpool3d_grad_with_argmax_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kMaxPool3DGradWithArgmaxGrad = "MaxPool3DGradWithArgmax";
constexpr size_t kInputShapeSize = 5;
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;

template <typename T, typename S>
bool MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *dy_addr = GetDeviceAddress<T>(inputs, kIndex1);
  S *index_addr = GetDeviceAddress<S>(inputs, kIndex2);
  T *dx_addr = GetDeviceAddress<T>(outputs, kIndex0);
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(dx_addr, 0, outputs[kIndex0]->size, reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'MaxPool3DWithArgmaxGrad' failed to cudaMemsetAsync");
  CalMaxPool3DGradWithArgmax(dy_addr, index_addr, x_dhw_, dy_dhw_, dy_ncdhw_, dx_addr, device_id_,
                             reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

bool MaxPool3DGradWithArgmaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
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

int MaxPool3DGradWithArgmaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
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

  const size_t nc_offset = 2;
  x_dhw_ = std::accumulate(x_shape.begin() + nc_offset, x_shape.end(), 1, std::multiplies<int64_t>());
  dy_dhw_ = std::accumulate(dy_shape.begin() + nc_offset, dy_shape.end(), 1, std::multiplies<int64_t>());
  dy_ncdhw_ = std::accumulate(dy_shape.begin(), dy_shape.end(), 1, std::multiplies<int64_t>());
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, MaxPool3DGradWithArgmaxGpuKernelMod::MaxPool3DArgmaxGradFunc>>
  MaxPool3DGradWithArgmaxGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<half, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<double, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint8_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint16_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint64_t, int32_t>},

    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &MaxPool3DGradWithArgmaxGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
};

std::vector<KernelAttr> MaxPool3DGradWithArgmaxGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MaxPool3DArgmaxGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MaxPool3DGradWithArgmax, []() {
  return std::make_shared<MaxPool3DGradWithArgmaxGpuKernelMod>(kMaxPool3DGradWithArgmaxGrad);
});
}  // namespace kernel
}  // namespace mindspore
