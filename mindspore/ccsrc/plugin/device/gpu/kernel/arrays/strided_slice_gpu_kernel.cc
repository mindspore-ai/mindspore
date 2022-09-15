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

#include "plugin/device/gpu/kernel/arrays/strided_slice_gpu_kernel.h"
#include <bitset>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t DynamicInputNum = 4;
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S = int64_t>
bool StridedSliceGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  StridedSlice(input_shape_, begin_, strides_, output_shape_, input, output,
               reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

bool StridedSliceGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int StridedSliceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  if (inputs.size() == DynamicInputNum) {
    is_dynamic_attr_ = true;
  }

  auto shape_signed = inputs[0]->GetShapeVector();
  input_shape_ = Convert2SizeTClipNeg(shape_signed);
  null_output_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input");
  if (null_output_) {
    return true;
  }
  if (input_shape_.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got " << input_shape_.size();
  }
  if (is_dynamic_attr_) {
    GetDynamicAttrIntValue(inputs, kBeginIndex_, inputsOnHost, kernel_name_, &begin_);
    GetDynamicAttrIntValue(inputs, kEndIndex_, inputsOnHost, kernel_name_, &end_);
    GetDynamicAttrIntValue(inputs, kStrideIndex_, inputsOnHost, kernel_name_, &strides_);
  }
  CollectInfo(base_operator, is_dynamic_attr_);

  return ret;
}

std::vector<std::pair<KernelAttr, StridedSliceGpuKernelMod::StridedSliceFunc>> StridedSliceGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &StridedSliceGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &StridedSliceGpuKernelMod::LaunchKernel<Complex<double>>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &StridedSliceGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &StridedSliceGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &StridedSliceGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &StridedSliceGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &StridedSliceGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &StridedSliceGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &StridedSliceGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &StridedSliceGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &StridedSliceGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &StridedSliceGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &StridedSliceGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &StridedSliceGpuKernelMod::LaunchKernel<bool>},

  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &StridedSliceGpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &StridedSliceGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &StridedSliceGpuKernelMod::LaunchKernel<half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &StridedSliceGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &StridedSliceGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   &StridedSliceGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   &StridedSliceGpuKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &StridedSliceGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt32),
   &StridedSliceGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt16),
   &StridedSliceGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   &StridedSliceGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   &StridedSliceGpuKernelMod::LaunchKernel<bool, int64_t>},
};

std::vector<KernelAttr> StridedSliceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, StridedSliceFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, StridedSlice, StridedSliceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
