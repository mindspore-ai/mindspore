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

#include "plugin/device/gpu/kernel/math/index_add_gpu_kernel.h"
#include <algorithm>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/index_add_impl.cuh"
#include "mindspore/core/ops/index_add.h"
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndexAddInputsNum = 3;
constexpr size_t kIndexAddOutputsNum = 1;
};  // namespace

bool IndexAddGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::IndexAdd>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast IndexAdd ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIndexAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIndexAddOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
  }
  t_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  kernel_func_ = func_list_[index].second;
  return true;
}

bool IndexAddGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  x_shape_ = inputs[kIndex0]->GetShapeVector();
  index_shape_ = inputs[kIndex1]->GetShapeVector();
  y_shape_ = inputs[kIndex2]->GetShapeVector();
  axis_value_ = GetValue<int64_t>(base_operator->GetAttr(kAttrAxis));
  auto it_x = std::find_if(x_shape_.begin(), x_shape_.end(), [](const int64_t sh) { return sh <= 0; });
  auto it_y = std::find_if(y_shape_.begin(), y_shape_.end(), [](const int64_t sh) { return sh <= 0; });
  auto it_idx = std::find_if(index_shape_.begin(), index_shape_.end(), [](const int64_t sh) { return sh <= 0; });
  if (it_x != x_shape_.end() || it_y != y_shape_.end() || it_idx != index_shape_.end()) {
    is_null_input_ = true;
    InitSizeLists();
    return true;
  }
  if (!CheckParams()) {
    return false;
  }

  outer_size_ = 1;
  for (size_t i = 0; i < axis_; ++i) {
    outer_size_ *= LongToSize(x_shape_[i]);
  }
  inner_size_ = 1;
  for (size_t i = axis_ + 1; i < x_shape_.size(); ++i) {
    inner_size_ *= LongToSize(x_shape_[i]);
  }
  x_axis_size_ = LongToSize(x_shape_[axis_]);
  y_axis_size_ = LongToSize(y_shape_[axis_]);

  x_size_ = t_size_;
  for (auto sh : x_shape_) {
    x_size_ *= LongToSize(sh);
  }
  index_size_ = sizeof(int32_t);
  for (auto sh : index_shape_) {
    index_size_ *= LongToSize(sh);
  }
  y_size_ = t_size_;
  for (auto sh : y_shape_) {
    y_size_ *= LongToSize(sh);
  }
  output_size_ = x_size_;

  InitSizeLists();
  return true;
}

void IndexAddGpuKernelMod::ResetResource() noexcept {
  x_size_ = 0;
  index_size_ = 0;
  y_size_ = 0;
  output_size_ = 0;
  inner_size_ = 0;
  outer_size_ = 0;
  x_axis_size_ = 0;
  y_axis_size_ = 0;
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void IndexAddGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(x_size_);
  input_size_list_.push_back(index_size_);
  input_size_list_.push_back(y_size_);
  output_size_list_.push_back(output_size_);
}

bool IndexAddGpuKernelMod::CheckParams() {
  // Check dimension(x) = dimension(y)
  if (x_shape_.size() != y_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'x' and 'y' should have the same dimension, but got "
                  << x_shape_.size() << " vs " << y_shape_.size();
    return false;
  }
  // Check dimension(indices) = 1
  if (index_shape_.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' should has one dimension, but got "
                  << index_shape_.size();
    return false;
  }
  // Check axis's value is valid
  auto x_rank = SizeToLong(x_shape_.size());
  if (axis_value_ < -x_rank || axis_value_ >= x_rank) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", 'axis' should be in range [" << -x_rank << ", " << x_rank
                  << "), but got " << axis_value_;
    return false;
  }
  if (axis_value_ < 0) {
    axis_value_ += x_rank;
  }
  axis_ = LongToSize(axis_value_);
  // Check indices's size = y.shape[axis]
  if (index_shape_[0] != y_shape_[axis_]) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << ", size of 'indices' should be same as size of 'y' in 'axis'th dimension, but got "
                  << index_shape_[0] << " vs " << y_shape_[axis_];
    return false;
  }
  // Check x.shape[i] = y.shape[i], except i = axis
  for (size_t i = 0; i < x_shape_.size(); ++i) {
    if (i != axis_ && x_shape_[i] != y_shape_[i]) {
      MS_LOG(ERROR)
        << "For '" << kernel_name_
        << ", the shape of 'x' and 'y' must be same except the 'axis'th dimension, but got different values: "
        << x_shape_[i] << " vs " << y_shape_[i] << " in dimension " << i;
      return false;
    }
  }
  return true;
}

template <typename T>
bool IndexAddGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIndexAddInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIndexAddOutputsNum, kernel_name_);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto *x = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto *index = reinterpret_cast<int *>(inputs[kIndex1]->addr);
  auto *y = reinterpret_cast<T *>(inputs[kIndex2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  CalIndexAdd(x, index, y, outer_size_, y_axis_size_, x_axis_size_, inner_size_, use_lock_, cuda_stream);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&output[0], &x[0], x_size_, cudaMemcpyDeviceToDevice, cuda_stream),
                                     "cudaMemcpyAsync output failed");
  return true;
}

std::vector<std::pair<KernelAttr, IndexAddGpuKernelMod::IndexAddFunc>> IndexAddGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &IndexAddGpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &IndexAddGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &IndexAddGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &IndexAddGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   &IndexAddGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   &IndexAddGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   &IndexAddGpuKernelMod::LaunchKernel<uint8_t>}};

std::vector<KernelAttr> IndexAddGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, IndexAddFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, IndexAdd, IndexAddGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
