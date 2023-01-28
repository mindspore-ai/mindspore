/**
 * Copyright 2022Huawei Technologies Co., Ltd
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
#include <iostream>
#include "plugin/device/gpu/kernel/arrays/search_sorted_gpu_kernal.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/search_sorted_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kSearchSortedInputsNum = 2;
constexpr int kSearchSortedOutputsNum = 1;
constexpr size_t kSearchSortedIndex0 = 0;
constexpr size_t kSearchSortedIndex1 = 1;
}  // namespace
bool SearchSortedGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::SearchSorted>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSearchSortedInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSearchSortedOutputsNum, kernel_name_);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  sequence_per_size_ = abstract::TypeIdSize(inputs[0]->GetDtype());
  value_per_size_ = abstract::TypeIdSize(inputs[1]->GetDtype());
  unit_output_size_ = abstract::TypeIdSize(outputs[0]->GetDtype());
  right = kernel_ptr_->get_right();
  return true;
}

int SearchSortedGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  auto sequence_shape = inputs.at(kSearchSortedIndex0)->GetShapeVector();
  auto value_shape = inputs.at(kSearchSortedIndex1)->GetShapeVector();
  (void)std::transform(sequence_shape.begin(), sequence_shape.end(), std::back_inserter(sequence_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  (void)std::transform(value_shape.begin(), value_shape.end(), std::back_inserter(value_shape_),
                       [](int64_t x) { return x < 0 ? 0 : LongToSize(x); });
  sequence_size_ = std::accumulate(sequence_shape_.begin(), sequence_shape_.end(), 1, std::multiplies<int64_t>());
  value_size_ = std::accumulate(value_shape_.begin(), value_shape_.end(), 1, std::multiplies<int64_t>());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  if (sequence_size_ == 0 || value_size_ == 0) {
    return KRET_UNKNOWN_SHAPE;
  }
  size_t output_size = output_elements_ * unit_output_size_;
  input_size_list_.push_back(sequence_size_ * sequence_per_size_);
  input_size_list_.push_back(value_size_ * value_per_size_);
  workspace_size_list_.push_back(sizeof(int));
  workspace_size_list_.push_back(sizeof(int));
  output_size_list_.push_back(output_size);
  return KRET_OK;
}

template <typename S, typename T>
bool SearchSortedGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  CheckParam<S, T>(inputs, outputs);
  auto sequence_ptr = GetDeviceAddress<S>(inputs, kSearchSortedIndex0);
  MS_EXCEPTION_IF_NULL(sequence_ptr);
  auto value_ptr = GetDeviceAddress<S>(inputs, kSearchSortedIndex1);
  MS_EXCEPTION_IF_NULL(value_ptr);
  int sequence_len1 = sequence_shape_.size();
  int value_len1 = value_shape_.size();
  if (sequence_len1 == 1) {
    should_last_repeat_ = False;
  }
  if (should_last_repeat_ && sequence_len1 != value_len1) {
    MS_EXCEPTION(ValueError)
      << "For '" << kernel_name_
      << "' sequence and value's dimemsion must be the same except the last dimension of 'values";
    return false;
  }
  if (value_len1 != 1) {
    for (int i = 0; i < sequence_len1 - 1; i++) {
      if (sequence_shape_[i] != value_shape_[i]) {
        MS_EXCEPTION(ValueError)
          << "For '" << kernel_name_
          << "' sequence and value's dimemsion must be the same except the last dimension of 'values";
        return false;
      }
    }
  }
  const S *sequence = GetDeviceAddress<S>(inputs, 0);
  const S *values = GetDeviceAddress<S>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  auto seq_dim = GetDeviceAddress<int>(workspace, 0);
  auto count1 = GetDeviceAddress<int>(workspace, 1);
  size_t input_elements_ = sequence_size_;
  size_t search_repeat = static_cast<size_t>(value_shape_.back());
  size_t search_len = static_cast<size_t>(sequence_shape_.back());
  if (!should_last_repeat_) {
    search_repeat = value_size_;
  }
  CalSearchSorted(input_elements_, sequence, values, output, seq_dim, search_repeat, search_len, right, device_id_,
                  cuda_stream_, count1);
  return true;
}

template <typename S, typename T>
void SearchSortedGpuKernelMod::CheckParam(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) {
  constexpr size_t kInputSize = 2;
  constexpr size_t kOutputSize = 1;
  if (inputs.size() != kInputSize) {
    MS_LOG(ERROR) << "Input number is: " << inputs.size() << ", but SearchSorted needs" << kInputSize << " inputs.";
  }
  if (outputs[0]->size / sizeof(T) != inputs[1]->size / sizeof(S)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of `v` and output must be equal, but got the dimension of `v` "
                  << inputs[1]->size << " and the dimension of output " << outputs[0]->size;
  }
  if (outputs.size() != kOutputSize) {
    MS_LOG(ERROR) << "Output number is " << outputs.size() << ", but SearchSorted needs " << kOutputSize << " outputs";
  }
  if (outputs[0]->size / sizeof(T) != inputs[1]->size / sizeof(S)) {
    MS_LOG(ERROR) << "The output dimensions " << outputs[0]->size << " must match the dimensions of input values "
                  << inputs[1]->size;
  }
}

std::vector<std::pair<KernelAttr, SearchSortedGpuKernelMod::SearchSortedFunc>> SearchSortedGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedGpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedGpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedGpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedGpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedGpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedGpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedGpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedGpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedGpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedGpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedGpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedGpuKernelMod::LaunchKernel<int8_t, int64_t>},
};

std::vector<KernelAttr> SearchSortedGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SearchSortedFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SearchSorted, SearchSortedGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
