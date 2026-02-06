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
#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include "kernel/gpu/cuda/arrays/argmaxandminwithvalue_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr char kArgMinWithValueOpName[] = "ArgMinWithValue";
constexpr char kArgMaxWithValueOpName[] = "ArgMaxWithValue";
constexpr char kMinDim[] = "MinDim";
constexpr char kMaxDim[] = "MaxDim";
}  // namespace

template <typename T, typename S>
bool ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, value_output_idx);
  S *index = GetDeviceAddress<S>(outputs, index_output_idx);
  auto status = CalGeneralReduction(small_, input, bound_, outer_size_, inner_size_, index, output,
                                    reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ArgMaxAndMinWithValueGpuKernelMod::ArgWithValueFunc>>
  ArgMaxAndMinWithValueGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &ArgMaxAndMinWithValueGpuKernelMod::LaunchKernel<half, int64_t>}};

std::vector<KernelAttr> ArgMaxAndMinWithValueGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  if (support_list.empty()) {
    (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, ArgWithValueFunc> &pair) { return pair.first; });
  }
  return support_list;
}

bool ArgMaxAndMinWithValueGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  // Check inputs and outputs size.
  if (inputs.size() != kInputNum) {
    MS_EXCEPTION(ArgumentError)
      << "For kernel mod[ArgMaxAndMinWithValueGpuKernelMod], the size of input should be 1, but got " << inputs.size();
  }
  if (outputs.size() != kOutputNum) {
    MS_EXCEPTION(ArgumentError)
      << "For kernel mod[ArgMaxAndMinWithValueGpuKernelMod], the size of output should be 2, but got "
      << outputs.size();
  }

  small_ = (kernel_name_ == kArgMinWithValueOpName || kernel_name_ == kMinDim) ? true : false;

  if (kernel_name_ == kMinDim || kernel_name_ == kMaxDim) {
    index_output_idx = 1;
    value_output_idx = 0;
  }

  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  kernel_func_ = func_list_[index].second;
  if (!is_match) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', it does not support this kernel data type: " << tensor_attr;
    return false;
  }
  return true;
}

bool ArgMaxAndMinWithValueGpuKernelMod::InitSize(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(inputs[0]);
  auto shape = Convert2SizeTClipNeg(inputs[0]->GetShapeVector());
  int64_t dims = SizeToLong(shape.size());
  axis_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  is_zero_dim_ = (dims == 0);

  if (axis_ < 0) {
    axis_ += dims;
  }
  size_t input_element_num = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
  if (input_element_num == 0) {
    return true;
  }

  bound_ = is_zero_dim_ ? 1 : shape[axis_];
  outer_size_ = 1;
  for (int64_t i = axis_ - 1; i >= 0; i--) {
    outer_size_ *= shape[i];
  }
  inner_size_ = 1;
  for (int64_t i = axis_ + 1; i < dims; i++) {
    inner_size_ *= shape[i];
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ArgMaxWithValue, ArgMaxAndMinWithValueGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ArgMinWithValue, ArgMaxAndMinWithValueGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
