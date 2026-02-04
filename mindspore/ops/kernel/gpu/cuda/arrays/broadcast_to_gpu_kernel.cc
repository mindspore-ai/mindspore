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

#include "kernel/gpu/cuda/arrays/broadcast_to_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda/math/broadcast_public.h"

namespace mindspore {
namespace kernel {
bool BroadcastToGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int BroadcastToGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  auto inp_shape = inputs[kIndex0]->GetShapeVector();
  auto out_shape = outputs[kIndex0]->GetShapeVector();
  if (inp_shape.size() > SHAPE_SIZE || out_shape.size() > SHAPE_SIZE) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output cannot be greater than "
                      << SHAPE_SIZE << ", but got the dimension of input: " << inp_shape.size()
                      << ", the dimension of output: " << out_shape.size();
  }

  is_null_input_ = CHECK_SHAPE_NULL(inp_shape, kernel_name_, "input");
  if (is_null_input_) {
    return KRET_OK;
  }
  SimplifyBroadcastToShape(inp_shape, out_shape, &simplified_inp_shape_, &simplified_out_shape_);
  is_broadcast_ = IsBinaryBroadcast(simplified_inp_shape_, simplified_out_shape_);
  return KRET_OK;
}

template <typename T>
bool BroadcastToGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  if (is_null_input_) {
    return true;
  }

  if (is_broadcast_) {
    BroadcastTo(simplified_inp_shape_, simplified_out_shape_, input_addr, output_addr, device_id_,
                reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    size_t cpy_size = SizeOf(simplified_out_shape_) * sizeof(T);
    cudaMemcpyAsync(output_addr, input_addr, cpy_size, cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  return true;
}

std::vector<std::pair<KernelAttr, BroadcastToGpuKernelMod::BroadcastToLaunchFunc>> BroadcastToGpuKernelMod::func_list_ =
  {
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
