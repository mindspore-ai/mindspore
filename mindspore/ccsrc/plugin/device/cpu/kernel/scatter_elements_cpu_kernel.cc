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

#include "plugin/device/cpu/kernel/scatter_elements_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore::kernel {
constexpr size_t kScatterElementsInputsNum = 3;
constexpr size_t kScatterElementsOutputsNum = 1;

namespace {
template <class T>
struct ReductionAdd {
  void operator()(T *a, const T &b) const { (*a) += b; }
};

template <class T>
struct ReductionAssignment {
  void operator()(T *a, const T &b) const { (*a) = b; }
};
}  // namespace

bool ScatterElementsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &others) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  if (!NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', resize failed.";
    return false;
  }
  kernel_name_ = base_operator->name();
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_dims_ = input_shape.size();
  if (input_dims_ < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x' should be greater than or equal to 1, but got " << input_dims_
                  << ".";
    return false;
  }
  indices_shape_ = inputs[kIndex1]->GetShapeVector();
  auto update_shape = inputs[kIndex2]->GetShapeVector();
  if (indices_shape_ != update_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'indice' and the shape of 'update' should be same, but got "
                  << "indice shape: " << indices_shape_ << "; "
                  << "update shape: " << update_shape << ".";
    return false;
  }
  if (input_dims_ != indices_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input_x', 'indice' and 'update' should be same, but got "
                  << "input_x dims: " << input_dims_ << "; "
                  << "indice dims: " << indices_shape_.size() << "; "
                  << "update dims: " << update_shape.size() << ".";
    return false;
  }

  if (base_operator->HasAttr(kAttrAxis)) {
    axis_ = GetValue<int64_t>(base_operator->GetAttr(kAttrAxis));
    if (axis_ < 0) {
      axis_ += input_dims_;
    }
  }

  if (axis_ >= static_cast<int64_t>(input_dims_) || axis_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the 'axis' should be less than input dims and greater than or equal 0, but got " << axis_
                  << ", while input dims is: " << input_dims_;
    return false;
  }

  for (size_t i = 0; i < input_dims_; ++i) {
    if (axis_ != static_cast<int64_t>(i) && input_shape[i] < indices_shape_[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the indices dims should be less than input dims, but got indice dim is: "
                    << indices_shape_[i] << " at axis: " << i << ", while input dim is:" << input_shape[i];
      return false;
    }
  }

  input_axis_size_ = SizeToInt(input_shape[axis_]);
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  indices_total_num_ =
    std::accumulate(indices_shape_.begin(), indices_shape_.end(), size_t(1), std::multiplies<size_t>());
  adjusted_indices_.resize(indices_total_num_);

  output_dim_stride_.resize(input_dims_);
  output_dim_stride_.back() = 1;
  for (int i = static_cast<int>(input_dims_ - 2); i >= 0; --i) {
    output_dim_stride_[i] = input_shape[i + 1] * output_dim_stride_[i + 1];
  }
  output_dim_index_.resize(input_dims_);
  output_dim_index_.assign(input_dims_, 0);
  return true;
}

template <typename S>
bool ScatterElementsCpuKernelMod::AdjustIndices(S *in_indices) {
  for (size_t i = 0; i < indices_total_num_; i++) {
    auto index = in_indices[i];
    if (index >= input_axis_size_ || index < -input_axis_size_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', index: " << index << " is expected to be within bounds ["
                    << -input_axis_size_ << ", " << input_axis_size_ << ")";
      return false;
    }
    if (index < 0) {
      index += input_axis_size_;
    }
    adjusted_indices_[i] = index;
  }
  return true;
}

size_t ScatterElementsCpuKernelMod::ComputeOutoutOffset(const int64_t &index) {
  size_t output_offset = 0;
  for (size_t i = 0; i < input_dims_; ++i) {
    if (static_cast<int64_t>(i) == axis_) {
      output_offset += index * output_dim_stride_[i];
    } else {
      output_offset += output_dim_index_[i] * output_dim_stride_[i];
    }
  }
  return output_offset;
}

void ScatterElementsCpuKernelMod::UpdateOutputDimIndex() {
  for (int i = static_cast<int>(input_dims_ - 1); i >= 0; --i) {
    auto cur = ++output_dim_index_[i];
    if (static_cast<int64_t>(cur) < indices_shape_[i]) {
      break;
    }
    output_dim_index_[i] = 0;
  }
  return;
}

template <typename T, typename ReductionT>
bool ScatterElementsCpuKernelMod::Scatter(const ReductionT &reduction_func, T *output, const T *updates) {
  for (size_t i = 0; i < indices_total_num_;) {
    auto index = adjusted_indices_[i];
    auto output_offset = ComputeOutoutOffset(index);
    reduction_func(output + output_offset, *(updates + i));
    if (++i == indices_total_num_) {
      break;
    }
    UpdateOutputDimIndex();
  }
  return true;
}

template <typename T, typename S, typename ReductionT>
bool ScatterElementsCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterElementsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterElementsOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *indices = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  auto *updates = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto bufferSize = outputs[kIndex0]->size;
  auto ret = memcpy_s(output, bufferSize, input, input_size_ * sizeof(T));
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', memory copy failed. Error no: " << ret;
    return false;
  }
  if (!AdjustIndices(indices)) {
    return false;
  }
  ReductionT reduction_func;
  return Scatter(reduction_func, output, updates);
}

std::map<std::string, std::vector<std::pair<KernelAttr, ScatterElementsCpuKernelMod::ScatterElementsLaunchFunc>>>
  ScatterElementsCpuKernelMod::func_map_ = {
    {kScatterElements,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt8)
         .AddOutputAttr(kNumberTypeInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<int8_t, int32_t, ReductionAssignment<int8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeUInt8)
         .AddOutputAttr(kNumberTypeUInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<uint8_t, int32_t, ReductionAssignment<uint8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &ScatterElementsCpuKernelMod::LaunchKernel<int32_t, int32_t, ReductionAssignment<int32_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &ScatterElementsCpuKernelMod::LaunchKernel<float16, int32_t, ReductionAssignment<float16>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &ScatterElementsCpuKernelMod::LaunchKernel<float, int32_t, ReductionAssignment<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &ScatterElementsCpuKernelMod::LaunchKernel<double, int32_t, ReductionAssignment<double>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &ScatterElementsCpuKernelMod::LaunchKernel<int64_t, int32_t, ReductionAssignment<int64_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt8)
         .AddOutputAttr(kNumberTypeInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<int8_t, int64_t, ReductionAssignment<int8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeUInt8)
         .AddOutputAttr(kNumberTypeUInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<uint8_t, int64_t, ReductionAssignment<uint8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &ScatterElementsCpuKernelMod::LaunchKernel<int32_t, int64_t, ReductionAssignment<int32_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &ScatterElementsCpuKernelMod::LaunchKernel<float16, int64_t, ReductionAssignment<float16>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &ScatterElementsCpuKernelMod::LaunchKernel<float, int64_t, ReductionAssignment<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &ScatterElementsCpuKernelMod::LaunchKernel<double, int64_t, ReductionAssignment<double>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &ScatterElementsCpuKernelMod::LaunchKernel<int64_t, int64_t, ReductionAssignment<int64_t>>}}},
    {kScatterAddWithAxis,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt8)
         .AddOutputAttr(kNumberTypeInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<int8_t, int32_t, ReductionAdd<int8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeUInt8)
         .AddOutputAttr(kNumberTypeUInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<uint8_t, int32_t, ReductionAdd<uint8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &ScatterElementsCpuKernelMod::LaunchKernel<int32_t, int32_t, ReductionAdd<int32_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &ScatterElementsCpuKernelMod::LaunchKernel<float16, int32_t, ReductionAdd<float16>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &ScatterElementsCpuKernelMod::LaunchKernel<float, int32_t, ReductionAdd<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &ScatterElementsCpuKernelMod::LaunchKernel<double, int32_t, ReductionAdd<double>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &ScatterElementsCpuKernelMod::LaunchKernel<int64_t, int32_t, ReductionAdd<int64_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt8)
         .AddOutputAttr(kNumberTypeInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<int8_t, int64_t, ReductionAdd<int8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeUInt8)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeUInt8)
         .AddOutputAttr(kNumberTypeUInt8),
       &ScatterElementsCpuKernelMod::LaunchKernel<uint8_t, int64_t, ReductionAdd<uint8_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &ScatterElementsCpuKernelMod::LaunchKernel<int32_t, int64_t, ReductionAdd<int32_t>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &ScatterElementsCpuKernelMod::LaunchKernel<float16, int64_t, ReductionAdd<float16>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &ScatterElementsCpuKernelMod::LaunchKernel<float, int64_t, ReductionAdd<float>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &ScatterElementsCpuKernelMod::LaunchKernel<double, int64_t, ReductionAdd<double>>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt64)
         .AddOutputAttr(kNumberTypeInt64),
       &ScatterElementsCpuKernelMod::LaunchKernel<int64_t, int64_t, ReductionAdd<int64_t>>}}}};

bool ScatterElementsCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(base_operator, false);
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(ERROR) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_map_[kernel_name_][pair.second].second;
  return true;
}

std::vector<KernelAttr> ScatterElementsCpuKernelMod::GetOpSupport() {
  auto iter = func_map_.find(kernel_type_);
  if (iter == func_map_.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ScatterElementsLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterElements,
                                 []() { return std::make_shared<ScatterElementsCpuKernelMod>(kScatterElements); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterAddWithAxis,
                                 []() { return std::make_shared<ScatterElementsCpuKernelMod>(kScatterAddWithAxis); });
}  // namespace mindspore::kernel
