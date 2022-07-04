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

#include "plugin/device/gpu/kernel/arrays/coalesce_gpu_kernel.h"
#include "mindspore/core/ops/coalesce.h"
#include <algorithm>

namespace mindspore {
namespace kernel {
constexpr size_t kCoalesceInputsNum = 3;
constexpr size_t kCoalesceOutputsNum = 3;
constexpr int kCOODimLimit = 2;

bool CoalesceGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Coalesce>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Coalesce ops failed!";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCoalesceInputsNum, kernel_ptr->name());
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCoalesceOutputsNum, kernel_ptr->name());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_ptr->name()
                      << "', it does not support this kernel data type: " << kernel_attr;
  }
  outputs_ = outputs;
  kernel_func_ = func_list_[index].second;
  is_need_retrieve_output_shape_ = true;
  return true;
}

int CoalesceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  outputs_ = outputs;
  input_indices_shape_ = inputs[kIndex0]->GetShapeVector();
  input_values_shape_ = inputs[kIndex1]->GetShapeVector();
  input_shape_shape_ = inputs[kIndex2]->GetShapeVector();
  if (!(CHECK_SHAPE_POSITIVE(input_indices_shape_) && CHECK_SHAPE_POSITIVE(input_values_shape_) &&
        CHECK_SHAPE_POSITIVE(input_shape_shape_))) {
    is_null_input_ = true;
    InitSizeLists();
    return 0;
  }

  MS_EXCEPTION_IF_CHECK_FAIL((kIndex1 < input_indices_shape_.size()), "Index is out of range.");
  MS_EXCEPTION_IF_CHECK_FAIL(!input_values_shape_.empty(), "input_values_shape_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!input_shape_shape_.empty(), "input_shape_shape_ should not be empty!");
  num_indices_ = input_indices_shape_[kIndex1];
  num_values_ = input_values_shape_[kIndex0];
  dims_ = input_shape_shape_[kIndex0];
  if (dims_ < kCOODimLimit) {
    MS_LOG_EXCEPTION << "For 'Coalesce', "
                     << "input_shape should have " << kCOODimLimit << " non-negative values, but got " << dims_
                     << "values!";
  }

  auto GetNums = [](const std::vector<int64_t> &shape) {
    size_t res = 1;
    for (const auto &sh : shape) {
      res *= LongToSize(sh);
    }
    return res;
  };
  input_indices_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype()) * GetNums(input_indices_shape_);
  input_values_size_ = abstract::TypeIdSize(inputs[kIndex1]->GetDtype()) * GetNums(input_values_shape_);
  input_shape_size_ = abstract::TypeIdSize(inputs[kIndex2]->GetDtype()) * GetNums(input_shape_shape_);
  output_indices_size_ = abstract::TypeIdSize(outputs[kIndex0]->GetDtype()) * GetNums(input_indices_shape_);
  output_values_size_ = abstract::TypeIdSize(outputs[kIndex1]->GetDtype()) * GetNums(input_values_shape_);
  output_shape_size_ = abstract::TypeIdSize(outputs[kIndex2]->GetDtype()) * GetNums(input_shape_shape_);
  InitSizeLists();
  return 0;
}

void CoalesceGpuKernelMod::ResetResource() noexcept {
  input_indices_size_ = 0;
  input_values_size_ = 0;
  input_shape_size_ = 0;
  output_indices_size_ = 0;
  output_values_size_ = 0;
  output_shape_size_ = 0;
  num_indices_ = 0;
  num_values_ = 0;
  dims_ = 0;
  post_output_size_ = 0;
  is_null_input_ = false;
  stream_ptr_ = nullptr;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void CoalesceGpuKernelMod::InitSizeLists() {
  (void)input_size_list_.emplace_back(input_indices_size_);
  (void)input_size_list_.emplace_back(input_values_size_);
  (void)input_size_list_.emplace_back(input_shape_size_);
  (void)output_size_list_.emplace_back(output_indices_size_);
  (void)output_size_list_.emplace_back(output_values_size_);
  (void)output_size_list_.emplace_back(output_shape_size_);
  (void)workspace_size_list_.emplace_back(input_indices_size_);
  (void)workspace_size_list_.emplace_back(input_indices_size_);
  (void)workspace_size_list_.emplace_back(input_indices_size_);
  (void)workspace_size_list_.emplace_back(input_indices_size_);
  (void)workspace_size_list_.emplace_back(input_indices_size_);
}

template <typename T, typename S, typename V>
bool CoalesceGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input_indices = GetDeviceAddress<T>(inputs, kIndex0);
  S *input_values = GetDeviceAddress<S>(inputs, kIndex1);
  V *input_shape = GetDeviceAddress<V>(inputs, kIndex2);

  T *output_indices = GetDeviceAddress<T>(outputs, kIndex0);
  S *output_values = GetDeviceAddress<S>(outputs, kIndex1);
  V *output_shape = GetDeviceAddress<V>(outputs, kIndex2);

  T *flatten_indices = GetDeviceAddress<T>(workspace, kIndex0);
  T *flatten_input_index = GetDeviceAddress<T>(workspace, kIndex1);
  T *flatten_sorted_index = GetDeviceAddress<T>(workspace, kIndex2);
  T *unique_indices = GetDeviceAddress<T>(workspace, kIndex3);
  T *idx = GetDeviceAddress<T>(workspace, kIndex4);

  stream_ptr_ = stream_ptr;
  // Flatten 2-d indices to 1-d
  FlattenIndices(input_indices, flatten_indices, input_shape, num_indices_,
                 reinterpret_cast<cudaStream_t>(stream_ptr_));
  // Unique indices
  post_output_size_ = CalUnique(flatten_indices, num_indices_, flatten_input_index, flatten_sorted_index,
                                unique_indices, idx, reinterpret_cast<cudaStream_t>(stream_ptr_));
  // Convert to 2-D indices
  ConvertTo2DIndices(unique_indices, output_indices, input_shape, post_output_size_,
                     reinterpret_cast<cudaStream_t>(stream_ptr_));

  // Coalesce values at the same index
  CalUniqueValues(idx, input_values, output_values, post_output_size_, num_values_,
                  reinterpret_cast<cudaStream_t>(stream_ptr_));

  // Copy shape
  device::gpu::CudaDriver::CopyDeviceMemToDeviceAsync(output_shape, input_shape, output_shape_size_,
                                                      reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}

void CoalesceGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                                     "For 'Coalesce', cudaStreamSynchronized failed");
  size_t output_num = outputs_.size();
  for (size_t i = 0; i < output_num; ++i) {
    std::vector<int64_t> shape = outputs_[i]->GetShapeVector();
    if (i == 0) {
      shape[1] = post_output_size_;
    } else if (i == 1) {
      shape[0] = post_output_size_;
    }
    outputs_[i]->SetShapeVector(std::vector<int64_t>(shape.begin(), shape.end()));
  }
}

std::vector<std::pair<KernelAttr, CoalesceGpuKernelMod::CoalesceFunc>> CoalesceGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt64),
   &CoalesceGpuKernelMod::LaunchKernel<int, half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64),
   &CoalesceGpuKernelMod::LaunchKernel<int, float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64),
   &CoalesceGpuKernelMod::LaunchKernel<int, double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeInt64),
   &CoalesceGpuKernelMod::LaunchKernel<int64_t, half, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeInt64),
   &CoalesceGpuKernelMod::LaunchKernel<int64_t, float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeInt64),
   &CoalesceGpuKernelMod::LaunchKernel<int64_t, double, int64_t>},
};
std::vector<KernelAttr> CoalesceGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CoalesceFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Coalesce, CoalesceGpuKernelMod);
};  // namespace kernel
}  // namespace mindspore
