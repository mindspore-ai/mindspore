/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/gatherv2_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
const size_t kStaticInputNum = 2;
const size_t kDynInputNum = 3;
constexpr char GATHER[] = "Gather";
constexpr char GATHERV2[] = "GatherV2";
constexpr char SPARSEGATHERV2[] = "SPARSEGATHERV2";
template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool GatherV2FwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  size_t input_num = inputs.size();
  if (input_num == kDynInputNum) {
    is_dynamic_shape_ = true;
    MS_LOG(INFO) << " GatherGpuV2FwdKernel running in Dynamic Mode.";
  } else if (input_num == kStaticInputNum) {
    axis_ = static_cast<int>(GetValue<int64_t>(base_operator->GetAttr("axis")));
    MS_LOG(INFO) << " GatherGpuV2FwdKernel running in Normal Mode.";
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  input_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  indices_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).first);
  if (is_dynamic_shape_) {
    axis_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).first);
  }
  return true;
}

int GatherV2FwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  input_shapes_ = inputs[kIndexZero]->GetShapeVector();
  indices_shapes_ = inputs[kIndexOne]->GetShapeVector();
  output_shapes_ = outputs[kIndexZero]->GetShapeVector();
  if (IsDynamic(input_shapes_) || IsDynamic(indices_shapes_) || IsDynamic(output_shapes_)) {
    return KRET_UNKNOWN_SHAPE;
  }
  is_null_input_ = CHECK_SHAPE_NULL(input_shapes_, kernel_name_, "input") ||
                   CHECK_SHAPE_NULL(indices_shapes_, kernel_name_, "indices") ||
                   CHECK_SHAPE_NULL(output_shapes_, kernel_name_, "output");
  if (is_null_input_) {
    InitSizeLists();
    return KRET_OK;
  }

  if (!is_dynamic_shape_) {
    int dims = SizeToInt(input_shapes_.size());
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << "," << dims
                        << "), but got " << axis_;
    }
    Reshape();
  }
  InitSizeLists();
  return KRET_OK;
}

std::vector<KernelAttr> GatherV2FwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const auto &pair) { return pair.first; });

  return support_list;
}

template <typename T, typename S, typename G>
bool GatherV2FwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);

  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  S *indices_addr = GetDeviceAddress<S>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  if (is_dynamic_shape_) {
    G *axis_device_address = GetDeviceAddress<G>(inputs, kIndex2);  // only get this if in dynamic mode
    G axis = 0;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&axis, axis_device_address, sizeof(G), cudaMemcpyDeviceToHost,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cudaMemcpy seq_lengths from device to host failed.");
    axis_ = static_cast<int>(axis);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed - GatherV2 - in dynamic mode");
    Reshape();
  }
  auto input_dim1 = input_shapes_[IntToSize(axis_)];

  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(indices_addr);
  GatherV2(input_addr, indices_addr, output_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2],
           LongToSize(input_dim1), reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, GatherV2FwdGpuKernelMod::GatherV2Func>> GatherV2FwdGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<float>, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<float>, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<double>, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex128),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<double>, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<double, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<double, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<float, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<float, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<half, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<half, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int64_t, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int64_t, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int16_t, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int16_t, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int8_t, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int8_t, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint64_t, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint64_t, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint16_t, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint16_t, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint8_t, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   &GatherV2FwdGpuKernelMod::LaunchKernel<uint8_t, int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &GatherV2FwdGpuKernelMod::LaunchKernel<bool, int, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &GatherV2FwdGpuKernelMod::LaunchKernel<bool, int64_t, int64_t>},
  // dynamic shape
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<double, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<double, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<double, int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<double, int, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<float, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<float, int, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<float, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<float, int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<half, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<half, int, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<half, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   &GatherV2FwdGpuKernelMod::LaunchKernel<half, int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   &GatherV2FwdGpuKernelMod::LaunchKernel<bool, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   &GatherV2FwdGpuKernelMod::LaunchKernel<bool, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeBool),
   &GatherV2FwdGpuKernelMod::LaunchKernel<bool, int, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeBool),
   &GatherV2FwdGpuKernelMod::LaunchKernel<bool, int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int, int, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &GatherV2FwdGpuKernelMod::LaunchKernel<int, int64_t, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<float>, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<float>, int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<double>, int, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &GatherV2FwdGpuKernelMod::LaunchKernel<Complex<double>, int64_t, int64_t>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Gather, GatherV2FwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
