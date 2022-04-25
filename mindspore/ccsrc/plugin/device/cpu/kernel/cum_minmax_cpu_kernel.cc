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

#include "plugin/device/cpu/kernel/cum_minmax_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumInputsNum = 1;
constexpr size_t kCumOutputsNum = 2;
template <typename T, typename S>
using CumMinMaxComputeFunc = std::function<std::pair<T, S>(const T &, const S &, const T &, const S &)>;

template <typename T, typename S, typename OP>
std::pair<T, S> cum_minmax(const T &a_val, const S &a_idx, const T &b_val, const S &b_idx) {
  OP op;
  if constexpr ((std::is_same_v<T, float>) || (std::is_same_v<T, double>)) {
    return std::isnan(a_val) || op(a_val, b_val) ? std::make_pair(a_val, a_idx) : std::make_pair(b_val, b_idx);
  } else if constexpr (std::is_same_v<T, float16>) {
    return isnan(a_val) || op(a_val, b_val) ? std::make_pair(a_val, a_idx) : std::make_pair(b_val, b_idx);
  }
  return op(a_val, b_val) ? std::make_pair(a_val, a_idx) : std::make_pair(b_val, b_idx);
}
}  // namespace

bool CumMinMaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << ", but got kernel name as " << kernel_name_;
  }

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  base_operator_ = base_operator;
  kernel_func_ = func_list_[kernel_type_][index].second;
  return true;
}

bool CumMinMaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &others) {
  if (!NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return false;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  auto rank = SizeToLong(input_shape.size());
  auto axis_input = GetValue<int64_t>(base_operator_->GetAttr(AXIS));
  auto axis = axis_input < 0 ? LongToSize(axis_input + rank) : LongToSize(axis_input);
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (i < axis) {
      outer_size_ *= input_shape.at(i);
    } else if (i > axis) {
      inner_size_ *= input_shape.at(i);
    } else {
      axis_size_ = input_shape.at(i);
    }
  }
  axis_inner_size_ = axis_size_ * inner_size_;
  element_size_ = axis_inner_size_ * outer_size_;
  return true;
}

size_t CumMinMaxCpuKernelMod::GetRealIndex(size_t index) {
  auto batch_idx = index / axis_size_;
  auto axis_idx = index - batch_idx * axis_size_;
  auto outer_idx = batch_idx / inner_size_;
  auto inner_idx = batch_idx - outer_idx * inner_size_;
  return outer_idx * axis_inner_size_ + axis_idx * inner_size_ + inner_idx;
}

template <typename T, typename S>
bool CumMinMaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumOutputsNum, kernel_name_);

  // Select the minimum/maximum computation function
  static const std::map<std::string, CumMinMaxComputeFunc<T, S>> cum_compute_func_map{
    {prim::kPrimCummax->name(), &cum_minmax<T, S, std::greater_equal<T>>},
    {prim::kPrimCummin->name(), &cum_minmax<T, S, std::less_equal<T>>},
  };
  if (cum_compute_func_map.find(kernel_name_) == cum_compute_func_map.end()) {
    MS_LOG(EXCEPTION) << "For 'CumMinMaxOp', the current kernel only support this operator in "
                      << Map2Str(cum_compute_func_map) << ", but got " << kernel_name_ << ".";
  }
  auto compute_func = cum_compute_func_map.at(kernel_name_);

  auto input_ptr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto value_ptr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto index_ptr = reinterpret_cast<S *>(outputs[kIndex1]->addr);

  // Cummin/Cummax parallel algorithm:
  // 1. Transpose the 'axis' dimension into the inner most dimension, [..., axis, ...] -> [..., axis]
  // 2. Flatten the transposed array and mark as [1, 0, 0, ..., 1, 0, ...]
  //    where 1 represents the start point., and 0 represents the non-start points.
  // 3. Then we can use multiple blocks way to deal with the whole arrays in parallel with three steps:
  //    (1) Divide blocks and process each block in parallel;
  //    (2) Merge the information of first element in each block sequentially;
  //    (3) Update each block in parallel;

  // Divide.
  std::vector<size_t> start_indices;
  std::mutex task_mutex;
  auto divide_task = [this, &compute_func, &input_ptr, &value_ptr, &index_ptr, &start_indices, &task_mutex](
                       size_t start, size_t end) {
    auto real_idx = GetRealIndex(start);
    auto pre_val = value_ptr[real_idx] = input_ptr[real_idx];
    auto pre_idx = index_ptr[real_idx] = start % axis_size_;
    for (size_t i = start + 1; i < end; i++) {
      auto idx = i % axis_size_;
      real_idx = GetRealIndex(i);
      auto val = input_ptr[real_idx];
      if (idx) {
        auto val_and_idx = compute_func(val, idx, pre_val, pre_idx);
        pre_val = value_ptr[real_idx] = val_and_idx.first;
        pre_idx = index_ptr[real_idx] = static_cast<S>(val_and_idx.second);
      } else {
        pre_val = value_ptr[real_idx] = input_ptr[real_idx];
        pre_idx = index_ptr[real_idx] = idx;
      }
    }
    std::lock_guard<std::mutex> task_lock(task_mutex);
    (void)start_indices.emplace_back(start);
  };
  CPUKernelUtils::ParallelFor(divide_task, element_size_, element_size_);

  // Merge.
  for (auto i : start_indices) {
    auto idx = i % axis_size_;
    if (idx) {
      auto real_idx = GetRealIndex(i);
      auto val = input_ptr[real_idx];
      auto real_pre_idx = GetRealIndex(i - 1);
      auto pre_val = value_ptr[real_pre_idx];
      auto val_and_idx = compute_func(val, idx, pre_val, idx - 1);
      value_ptr[real_idx] = val_and_idx.first;
      index_ptr[real_idx] = static_cast<S>(val_and_idx.second);
    }
  }

  // Update.
  auto update_task = [this, &compute_func, &input_ptr, &value_ptr, &index_ptr](size_t start, size_t end) {
    auto real_idx = GetRealIndex(start);
    auto pre_val = value_ptr[real_idx];
    auto pre_idx = index_ptr[real_idx];
    for (size_t i = start; i < end; i++) {
      auto idx = i % axis_size_;
      if (idx) {
        real_idx = GetRealIndex(i);
        auto val = input_ptr[real_idx];
        auto val_and_idx = compute_func(val, idx, pre_val, pre_idx);
        pre_val = value_ptr[real_idx] = val_and_idx.first;
        pre_idx = index_ptr[real_idx] = static_cast<S>(val_and_idx.second);
      } else {
        break;
      }
    }
  };
  CPUKernelUtils::ParallelFor(update_task, element_size_, element_size_);
  return true;
}

// Note that in definition of primitive, Cummin return int32 as indices and Cummax return int64 as indices. (see
// cummax.cc and cummin.cc).
std::map<std::string, std::vector<std::pair<KernelAttr, CumMinMaxCpuKernelMod::CumMinMaxLaunchFunc>>>
  CumMinMaxCpuKernelMod::func_list_ = {
    {kCummin,
     {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<int8_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<int16_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<int32_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<int64_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint8_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint16_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint32_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint64_t, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<float16, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<float, int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
       &CumMinMaxCpuKernelMod::LaunchKernel<double, int32_t>}}},
    {kCummax,
     {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<int8_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<int16_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<int32_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<int64_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint8_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint16_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint32_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<uint64_t, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<float16, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<float, int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
       &CumMinMaxCpuKernelMod::LaunchKernel<double, int64_t>}}}};

std::vector<KernelAttr> CumMinMaxCpuKernelMod::GetOpSupport() {
  auto iter = func_list_.find(kernel_type_);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "Cum_minmax cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(
    iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CumMinMaxCpuKernelMod::CumMinMaxLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_WITH_NAME_PARAM(NativeCpuKernelMod, Cummin, CumMinMaxCpuKernelMod);
MS_KERNEL_FACTORY_REG_WITH_NAME_PARAM(NativeCpuKernelMod, Cummax, CumMinMaxCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
