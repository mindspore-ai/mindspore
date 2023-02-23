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

#include "plugin/device/gpu/kernel/nn/adaptive_max_pool3d_gpu_kernel.h"
#include <memory>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool3d_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kAdaptiveMaxPool3D = "AdaptiveMaxPool3D";
constexpr size_t kInputsNum = 2;
constexpr size_t kMinInputShapeSize = 4;
constexpr size_t kDimNum4 = 4;
constexpr size_t kDimNum5 = 5;

template <typename T>
bool AdaptiveMaxPool3DKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                              const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  auto *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  auto *output_size_addr = GetDeviceAddress<int32_t>(inputs, kIndex1);
  auto *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto *mask_addr = GetDeviceAddress<int32_t>(outputs, kIndex1);

  auto status = ApplyAdaptiveMaxPool3D(output_size_, channels_, input_depth_, input_height_, input_width_, input_addr,
                                       output_size_addr, output_addr, mask_addr, device_id_,
                                       reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

bool AdaptiveMaxPool3DKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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

int AdaptiveMaxPool3DKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  if (inputs.size() != kInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << inputs.size();
    return KRET_RESIZE_FAILED;
  }

  ShapeVector input_shape = inputs.at(kIndex0)->GetShapeVector();
  ShapeVector output_shape = outputs.at(kIndex0)->GetShapeVector();
  const size_t dim_num = input_shape.size();
  if (!(dim_num == kDimNum4 || dim_num == kDimNum5)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input 'x' dimensions should be equal to 4 or 5, but got "
                  << dim_num;
    return KRET_RESIZE_FAILED;
  }

  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_RESIZE_FAILED;
  }

  const size_t kIndexC = dim_num - kIndex4;
  const size_t kIndexD = dim_num - kIndex3;
  const size_t kIndexH = dim_num - kIndex2;
  const size_t kIndexW = dim_num - kIndex1;
  channels_ = input_shape[kIndexC];
  input_depth_ = input_shape[kIndexD];
  input_height_ = input_shape[kIndexH];
  input_width_ = input_shape[kIndexW];

  output_size_ = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_size_ *= output_shape[i];
  }

  return KRET_OK;
}

std::vector<std::pair<KernelAttr, AdaptiveMaxPool3DKernelMod::AdaptiveMaxPool3DFunc>>
  AdaptiveMaxPool3DKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &AdaptiveMaxPool3DKernelMod::LaunchKernel<uint64_t>},
};

std::vector<KernelAttr> AdaptiveMaxPool3DKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AdaptiveMaxPool3DFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AdaptiveMaxPool3D,
                                 []() { return std::make_shared<AdaptiveMaxPool3DKernelMod>(kAdaptiveMaxPool3D); });
}  // namespace kernel
}  // namespace mindspore
