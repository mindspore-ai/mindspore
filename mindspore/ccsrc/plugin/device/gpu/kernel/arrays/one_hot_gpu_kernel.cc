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

#include "plugin/device/gpu/kernel/arrays/one_hot_gpu_kernel.h"
#include <cstdint>
#include "mindspore/core/ops/one_hot.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/one_hot_impl.cuh"

namespace mindspore {
namespace kernel {
bool OneHotGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t min_input_num = 3;
  constexpr size_t max_input_num = 4;
  if (inputs.size() != min_input_num && inputs.size() != max_input_num) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input num should be 3 or 4, but get: " << inputs.size();
    return false;
  }
  if (inputs.size() == max_input_num) {
    is_dynamic_shape_ = true;
  }
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int OneHotGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto output_shape = LongVecToSizeVec(outputs[kIndex0]->GetShapeVector());
  auto one_hot_ptr = std::dynamic_pointer_cast<ops::OneHot>(base_operator);
  MS_EXCEPTION_IF_NULL(one_hot_ptr);
  int64_t axis = one_hot_ptr->get_axis();

  int64_t input_dims = static_cast<int64_t>(input_shape.size());
  int64_t output_dims = static_cast<int64_t>(output_shape.size());
  if (axis > input_dims || axis >= output_dims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' must be less than the dimension of input and output"
                  << ", but got 'axis': " << axis << ", the dimension of input: " << input_dims
                  << ", the dimension of output: " << output_dims;
    return KRET_RESIZE_FAILED;
  }
  const int64_t default_axis = -1;
  left_dim_size_ = 1;
  right_dim_size_ = 1;

  // Compress arbitrary tensor dimensions into three dimensions (left_dims, depth, right_dims).
  for (size_t i = 0; i < input_shape.size(); i++) {
    auto dim_size = input_shape[i];
    if (axis == default_axis || i < IntToSize(axis)) {
      left_dim_size_ *= dim_size;
    }
    if (axis != default_axis && i >= IntToSize(axis)) {
      right_dim_size_ *= dim_size;
    }
  }
  if (axis == default_axis) {
    depth_ = output_shape[output_shape.size() - 1];
  } else {
    depth_ = output_shape[IntToSize(axis)];
  }
  return KRET_OK;
}

template <typename T, typename S, typename G = int>
bool OneHotGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  size_t on_value_idx = 1;
  size_t off_value_idx = 2;
  if (is_dynamic_shape_) {
    on_value_idx++;
    off_value_idx++;
  }
  const S *indices = GetDeviceAddress<S>(inputs, 0);
  const T *on_value = GetDeviceAddress<T>(inputs, on_value_idx);
  const T *off_value = GetDeviceAddress<T>(inputs, off_value_idx);
  T *output = GetDeviceAddress<T>(outputs, 0);
  OneHot(indices, depth_, on_value, off_value, left_dim_size_, right_dim_size_, output,
         reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, OneHotGpuKernelMod::OneHotLaunchFunc>> OneHotGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &OneHotGpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &OneHotGpuKernelMod::LaunchKernel<half, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &OneHotGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &OneHotGpuKernelMod::LaunchKernel<half, int64_t>},
  // dynamic shape
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &OneHotGpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &OneHotGpuKernelMod::LaunchKernel<half, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &OneHotGpuKernelMod::LaunchKernel<float, int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &OneHotGpuKernelMod::LaunchKernel<half, int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &OneHotGpuKernelMod::LaunchKernel<float, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &OneHotGpuKernelMod::LaunchKernel<half, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &OneHotGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &OneHotGpuKernelMod::LaunchKernel<half, int64_t>},
};

std::vector<KernelAttr> OneHotGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, OneHotGpuKernelMod::OneHotLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, OneHot, OneHotGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
