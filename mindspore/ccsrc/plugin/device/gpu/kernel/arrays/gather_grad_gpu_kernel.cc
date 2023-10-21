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

#include "plugin/device/gpu/kernel/arrays/gather_grad_gpu_kernel.h"
#include "mindspore/core/ops/array_ops.h"
#include "kernel/kernel_get_value.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/hal/device/gpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

namespace {
#define V2REGISTER(X1, X2, X3, X4, OUTPUT, T1, T2)                                                          \
  {                                                                                                         \
    KernelAttr().AddInputAttr(X1).AddInputAttr(X2).AddInputAttr(X3).AddInputAttr(X4).AddOutputAttr(OUTPUT), \
      &GatherGradGpuKernelMod::LaunchKernel<T1, T2>                                                         \
  }

constexpr auto kGatherDGrad = "GatherDGrad";
constexpr auto kGatherDGradV2 = "GatherDGradV2";
constexpr size_t kStaticSize = 2;
constexpr size_t kDynamicSize = 4;
constexpr size_t kDynamicDimIdx = 1;
constexpr size_t kDynamicIndexIdx = 2;
constexpr size_t kDynamicGradIdx = 3;
}  // namespace

bool GatherGradGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = stream_ptr;
  return kernel_func_(this, inputs, outputs, cuda_stream_);
}

template <typename T, typename S>
bool GatherGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                                          void *stream_ptr) {
  T *index_addr = GetDeviceAddress<T>(inputs, index_idx_);
  S *grad_addr = GetDeviceAddress<S>(inputs, grad_idx_);
  S *output_addr = GetDeviceAddress<S>(outputs, 0);

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(output_addr, 0, outputs[kIndex0]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
    "GatherGrad cudaMemSet Failed");

  auto status = GatherGrad(index_addr, grad_addr, output_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2],
                           dims_[kIndex3], reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

bool GatherGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  size_t input_num = inputs.size();
  if (input_num == kStaticSize) {
    index_idx_ = 0;
    grad_idx_ = 1;
  } else if (input_num == kDynamicSize) {
    dim_idx_ = kDynamicDimIdx;
    index_idx_ = kDynamicIndexIdx;
    grad_idx_ = kDynamicGradIdx;
    is_v2_ = true;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 4, but got " << input_num;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;
  idx_type_size_ = abstract::TypeIdSize(inputs.at(index_idx_)->GetDtype());
  grad_type_size_ = abstract::TypeIdSize(inputs.at(grad_idx_)->GetDtype());
  if (is_v2_) {
    dim_type_ = inputs.at(dim_idx_)->GetDtype();
  }
  return true;
}

int GatherGradGpuKernelMod::GetGatherDGradDimValue(const BaseOperatorPtr &base_operator) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::GatherDGrad>(base_operator);
  return static_cast<int>(kernel_ptr->get_dim());
}

void GatherGradGpuKernelMod::CalculateDim(int axis) {
  if (grad_shapes_.size() != index_shapes_.size() || grad_shapes_.size() != output_shapes_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of grad, index and output must be the same, but got the dimension of "
                      << "grad: " << grad_shapes_.size() << ", the dimension of index: " << index_shapes_.size()
                      << ", the dimension of output: " << output_shapes_.size();
  }
  int dims = SizeToInt(grad_shapes_.size());
  if (axis < -dims || axis >= dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << "," << dims
                      << "), but got " << axis;
  }
  if (axis < 0) {
    axis += dims;
  }
  int64_t dim_before_axis = 1;
  for (size_t i = 0; i < IntToSize(axis); i++) {
    dim_before_axis *= output_shapes_[i];
  }
  size_t dim_at_axis_index = LongToSizeClipNeg(index_shapes_[IntToSize(axis)]);
  size_t dim_at_axis_output = LongToSizeClipNeg(output_shapes_[IntToSize(axis)]);
  int64_t dim_after_axis = 1;
  for (size_t i = IntToSize(axis) + 1; i < output_shapes_.size(); i++) {
    dim_after_axis *= output_shapes_[i];
  }
  dims_[kIndex0] = LongToSize(dim_before_axis);
  dims_[kIndex1] = dim_at_axis_index;
  dims_[kIndex2] = dim_at_axis_output;
  dims_[kIndex3] = LongToSize(dim_after_axis);
}

int GatherGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  index_shapes_ = inputs.at(index_idx_)->GetShapeVector();
  output_shapes_ = outputs.at(kIndex0)->GetShapeVector();
  grad_shapes_ = inputs.at(grad_idx_)->GetShapeVector();
  if (is_v2_) {
    int64_t dim = 0;
    if (!TryGetIntValue(inputs, kIndex1, kernel_name_, &dim)) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', cant get axis.";
    }
    CalculateDim(dim);
    dim_shapes_ = inputs.at(dim_idx_)->GetShapeVector();
  } else {
    auto dim = GetGatherDGradDimValue(base_operator);
    CalculateDim(dim);
  }
  return KRET_OK;
}

std::vector<KernelAttr> GatherGradGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "For 'FloatStatus op', the kernel name must be in "
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, GatherGradGpuKernelMod::GatherGradOpFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, GatherGradOpFunc> &item) { return item.first; });
  return support_list;
}

std::map<std::string, std::vector<std::pair<KernelAttr, GatherGradGpuKernelMod::GatherGradOpFunc>>>
  GatherGradGpuKernelMod::kernel_attr_map_ = {
    {kGatherDGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &GatherGradGpuKernelMod::LaunchKernel<int, double>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, double>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &GatherGradGpuKernelMod::LaunchKernel<int, float>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, float>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &GatherGradGpuKernelMod::LaunchKernel<int, half>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, half>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int, int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int, int8_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int8_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &GatherGradGpuKernelMod::LaunchKernel<int, int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &GatherGradGpuKernelMod::LaunchKernel<int, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int, uchar>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, uchar>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int, uint>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, uint>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &GatherGradGpuKernelMod::LaunchKernel<int, bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int, uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, uint32_t>}}},
    {kGatherDGradV2,
     {V2REGISTER(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeComplex128,
                 kNumberTypeComplex128, int, Complex<double>),
      V2REGISTER(kNumberTypeComplex128, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeComplex128,
                 kNumberTypeComplex128, int64_t, Complex<double>),
      V2REGISTER(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeComplex64, kNumberTypeComplex64,
                 int, Complex<float>),
      V2REGISTER(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeComplex64, kNumberTypeComplex64,
                 int64_t, Complex<float>),
      V2REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64, int,
                 double),
      V2REGISTER(kNumberTypeFloat64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                 int64_t, double),
      V2REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, int,
                 float),
      V2REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                 int64_t, float),
      V2REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16, int,
                 half),
      V2REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                 int64_t, half),
      V2REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int,
                 int64_t),
      V2REGISTER(kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                 int64_t),
      V2REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int, int),
      V2REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int64_t,
                 int),
      V2REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16, int,
                 int16_t),
      V2REGISTER(kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16, int64_t,
                 int16_t),
      V2REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int, int8_t),
      V2REGISTER(kNumberTypeInt8, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int, int8_t),
      V2REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeUInt64, int,
                 uint64_t),
      V2REGISTER(kNumberTypeUInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeUInt64, int64_t,
                 uint64_t),
      V2REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeUInt32, int,
                 uint32_t),
      V2REGISTER(kNumberTypeUInt32, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeUInt32, int64_t,
                 uint32_t),
      V2REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeUInt16, int,
                 uint16_t),
      V2REGISTER(kNumberTypeUInt16, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeUInt16, int64_t,
                 uint16_t),
      V2REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, int, uchar),
      V2REGISTER(kNumberTypeUInt8, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, int64_t,
                 uchar),
      V2REGISTER(kNumberTypeBool, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeBool, kNumberTypeBool, int, bool),
      V2REGISTER(kNumberTypeBool, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool, kNumberTypeBool, int64_t, bool),
      V2REGISTER(kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeComplex128,
                 kNumberTypeComplex128, int, Complex<double>),
      V2REGISTER(kNumberTypeComplex128, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeComplex128,
                 kNumberTypeComplex128, int64_t, Complex<double>),
      V2REGISTER(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeComplex64, kNumberTypeComplex64,
                 int, Complex<float>),
      V2REGISTER(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeComplex64, kNumberTypeComplex64,
                 int64_t, Complex<float>),
      V2REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeFloat64, kNumberTypeFloat64, int,
                 double),
      V2REGISTER(kNumberTypeFloat64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64,
                 int64_t, double),
      V2REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, int,
                 float),
      V2REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32,
                 int64_t, float),
      V2REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16, int,
                 half),
      V2REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16,
                 int64_t, half),
      V2REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, int,
                 int64_t),
      V2REGISTER(kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, int64_t,
                 int64_t),
      V2REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int, int),
      V2REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int64_t,
                 int),
      V2REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt16, kNumberTypeInt16, int,
                 int16_t),
      V2REGISTER(kNumberTypeInt16, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt16, kNumberTypeInt16, int64_t,
                 int16_t),
      V2REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt8, kNumberTypeInt8, int, int8_t),
      V2REGISTER(kNumberTypeInt8, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt8, kNumberTypeInt8, int, int8_t),
      V2REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeUInt64, kNumberTypeUInt64, int,
                 uint64_t),
      V2REGISTER(kNumberTypeUInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeUInt64, kNumberTypeUInt64, int64_t,
                 uint64_t),
      V2REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeUInt32, kNumberTypeUInt32, int,
                 uint32_t),
      V2REGISTER(kNumberTypeUInt32, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeUInt32, kNumberTypeUInt32, int64_t,
                 uint32_t),
      V2REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeUInt16, kNumberTypeUInt16, int,
                 uint16_t),
      V2REGISTER(kNumberTypeUInt16, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeUInt16, kNumberTypeUInt16, int64_t,
                 uint16_t),
      V2REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt8, int, uchar),
      V2REGISTER(kNumberTypeUInt8, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeUInt8, kNumberTypeUInt8, int64_t,
                 uchar),
      V2REGISTER(kNumberTypeBool, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeBool, kNumberTypeBool, int, bool),
      V2REGISTER(kNumberTypeBool, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeBool, kNumberTypeBool, int64_t,
                 bool)}}};
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, GatherDGrad,
                                 []() { return std::make_shared<GatherGradGpuKernelMod>(kGatherDGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, GatherDGradV2,
                                 []() { return std::make_shared<GatherGradGpuKernelMod>(kGatherDGradV2); });
}  // namespace kernel
}  // namespace mindspore
