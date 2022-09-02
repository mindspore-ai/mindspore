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

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kGatherDGrad = "GatherDGrad";
constexpr auto kGatherDGradV2 = "GatherDGradV2";
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

  GatherGrad(index_addr, grad_addr, output_addr, dims_[kIndex0], dims_[kIndex1], dims_[kIndex2], dims_[kIndex3],
             reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

bool GatherGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  size_t input_num = inputs.size();
  constexpr size_t kStaticSize = 2;
  constexpr size_t kDynamicSize = 3;
  if (input_num == kStaticSize) {
    index_idx_ = 0;
    grad_idx_ = 1;
  } else if (input_num == kDynamicSize) {
    index_idx_ = 1;
    constexpr size_t kDynamicGradIdx = 2;
    grad_idx_ = kDynamicGradIdx;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
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

  return true;
}

int GatherGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KRET_OK;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();

  index_shapes_ = inputs.at(index_idx_)->GetShapeVector();
  if (!IsValidShape(index_shapes_)) {
    ret = KRET_UNKNOWN_SHAPE;
    input_size_list_.push_back(idx_type_size_);
  } else {
    input_size_list_.push_back(idx_type_size_ * SizeOf(index_shapes_));
  }

  grad_shapes_ = inputs.at(grad_idx_)->GetShapeVector();
  if (!IsValidShape(grad_shapes_)) {
    ret = (ret == KRET_OK ? KRET_UNKNOWN_OUT_SHAPE : ret);
    input_size_list_.push_back(grad_type_size_);
  } else {
    input_size_list_.push_back(grad_type_size_ * SizeOf(grad_shapes_));
  }

  output_shapes_ = outputs.at(kIndex0)->GetShapeVector();
  if (!IsValidShape(output_shapes_)) {
    ret = (ret == KRET_OK ? KRET_UNKNOWN_OUT_SHAPE : ret);
    output_size_list_.push_back(grad_type_size_);
  } else {
    output_size_list_.push_back(grad_type_size_ * SizeOf(output_shapes_));
  }

  if (ret == KRET_OK) {
    if (grad_shapes_.size() != index_shapes_.size() || grad_shapes_.size() != output_shapes_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of grad, index and output must be the same, but got the dimension of "
                        << "grad: " << grad_shapes_.size() << ", the dimension of index: " << index_shapes_.size()
                        << ", the dimension of output: " << output_shapes_.size();
    }
    int dims = SizeToInt(grad_shapes_.size());
    MS_EXCEPTION_IF_NULL(base_operator);

    size_t input_num = inputs.size();
    constexpr size_t kStaticSize = 2;
    constexpr size_t kDynamicSize = 3;
    if (input_num == kStaticSize) {
      auto kernel_ptr = std::dynamic_pointer_cast<ops::GatherDGrad>(base_operator);
      axis_ = static_cast<int>(kernel_ptr->get_dim());
    } else if (input_num == kDynamicSize) {
      auto kernel_ptr = std::dynamic_pointer_cast<ops::GatherDGradV2>(base_operator);
      axis_ = static_cast<int>(kernel_ptr->get_dim());
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
    }

    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << "," << dims
                        << "), but got " << axis_;
    }
    if (axis_ < 0) {
      axis_ += dims;
    }

    int64_t dim_before_axis = 1;
    for (size_t i = 0; i < IntToSize(axis_); i++) {
      dim_before_axis *= output_shapes_[i];
    }
    size_t dim_at_axis_index = LongToSizeClipNeg(index_shapes_[IntToSize(axis_)]);
    size_t dim_at_axis_output = LongToSizeClipNeg(output_shapes_[IntToSize(axis_)]);
    int64_t dim_after_axis = 1;
    for (size_t i = IntToSize(axis_) + 1; i < output_shapes_.size(); i++) {
      dim_after_axis *= output_shapes_[i];
    }

    dims_[kIndex0] = LongToSize(dim_before_axis);
    dims_[kIndex1] = dim_at_axis_index;
    dims_[kIndex2] = dim_at_axis_output;
    dims_[kIndex3] = LongToSize(dim_after_axis);
  }

  return static_cast<int>(ret);
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
     {{KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &GatherGradGpuKernelMod::LaunchKernel<int, double>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, double>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &GatherGradGpuKernelMod::LaunchKernel<int, float>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, float>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &GatherGradGpuKernelMod::LaunchKernel<int, half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int, int>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt8)
         .AddOutputAttr(kNumberTypeInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int, int8_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt8)
         .AddOutputAttr(kNumberTypeInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int8_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt16)
         .AddOutputAttr(kNumberTypeInt16),
       &GatherGradGpuKernelMod::LaunchKernel<int, int16_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt16)
         .AddOutputAttr(kNumberTypeInt16),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int16_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &GatherGradGpuKernelMod::LaunchKernel<int, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, int64_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeUInt8)
         .AddOutputAttr(kNumberTypeUInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int, uchar>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeUInt8)
         .AddOutputAttr(kNumberTypeUInt8),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, uchar>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeUInt32)
         .AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int, uint>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeUInt32)
         .AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, uint>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeBool)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeBool)
         .AddOutputAttr(kNumberTypeBool),
       &GatherGradGpuKernelMod::LaunchKernel<int, bool>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeBool)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeBool)
         .AddOutputAttr(kNumberTypeBool),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, bool>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeUInt32)
         .AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int, uint32_t>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeUInt32)
         .AddOutputAttr(kNumberTypeUInt32),
       &GatherGradGpuKernelMod::LaunchKernel<int64_t, uint32_t>}}}};
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, GatherDGrad,
                                 []() { return std::make_shared<GatherGradGpuKernelMod>(kGatherDGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, GatherDGradV2,
                                 []() { return std::make_shared<GatherGradGpuKernelMod>(kGatherDGradV2); });
}  // namespace kernel
}  // namespace mindspore
