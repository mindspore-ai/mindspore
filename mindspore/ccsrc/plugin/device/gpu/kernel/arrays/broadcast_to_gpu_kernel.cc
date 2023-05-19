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

#include "plugin/device/gpu/kernel/arrays/broadcast_to_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool BroadcastToGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  return true;
}
void BroadcastToGpuKernelMod::ResetResource() noexcept {
  input_size_ = 1;
  output_size_ = 1;
  for (size_t i = 0; i < SHAPE_SIZE; ++i) {
    input_shape_[i] = 1;
    output_shape_[i] = 1;
  }
  is_null_input_ = false;
}

int BroadcastToGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  auto input_shapes = inputs[kIndex0]->GetShapeVector();
  auto output_shapes = outputs[kIndex0]->GetShapeVector();

  auto it_x = std::find_if(input_shapes.begin(), input_shapes.end(), [](int64_t sh) { return sh < 0; });
  if (it_x != input_shapes.end()) {
    return KRET_UNKNOWN_SHAPE;
  }

  if (input_shapes.size() > SHAPE_SIZE || output_shapes.size() > SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output cannot be greater than "
                      << SHAPE_SIZE << ", but got the dimension of input: " << input_shapes.size()
                      << ", the dimension of output: " << output_shapes.size();
  }

  if (output_shapes.size() < input_shapes.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output cannot be less than the dimension of input "
                      << ", but got the dimension of input: " << input_shapes.size()
                      << ", the dimension of output: " << output_shapes.size();
  }
  size_t offset = output_shapes.size() - input_shapes.size();
  for (size_t i = 0; i < input_shapes.size(); i++) {
    input_shape_[i + offset] = LongToSizeClipNeg(input_shapes[i]);
  }

  for (size_t j = 0; j < output_shapes.size(); j++) {
    output_shape_[j] = LongToSizeClipNeg(output_shapes[j]);
  }

  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies{});
  output_size_ = std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies{});

  input_size_list_.clear();
  output_size_list_.clear();
  input_size_list_.push_back(input_size_ * input_type_size_);
  output_size_list_.push_back(output_size_ * input_type_size_);
  return KRET_OK;
}

template <typename T>
bool BroadcastToGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  BroadcastTo(input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3], input_shape_[4], input_shape_[5],
              input_shape_[6], input_shape_[7], output_shape_[0], output_shape_[1], output_shape_[2], output_shape_[3],
              output_shape_[4], output_shape_[5], output_shape_[6], output_shape_[7], input_addr, output_addr,
              reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, BroadcastToGpuKernelMod::BroadcastToLaunchFunc>> BroadcastToGpuKernelMod::func_list_ =
  {
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &BroadcastToGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &BroadcastToGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &BroadcastToGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &BroadcastToGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<double>>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &BroadcastToGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &BroadcastToGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &BroadcastToGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     &BroadcastToGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<double>>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &BroadcastToGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &BroadcastToGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &BroadcastToGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     &BroadcastToGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<double>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &BroadcastToGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &BroadcastToGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &BroadcastToGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeUInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeBool),
     &BroadcastToGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex64),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<double>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &BroadcastToGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &BroadcastToGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &BroadcastToGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &BroadcastToGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &BroadcastToGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &BroadcastToGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &BroadcastToGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool),
     &BroadcastToGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &BroadcastToGpuKernelMod::LaunchKernel<utils::Complex<double>>},
};

std::vector<KernelAttr> BroadcastToGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, BroadcastToGpuKernelMod::BroadcastToLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BroadcastTo, BroadcastToGpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DynamicBroadcastTo, BroadcastToGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
