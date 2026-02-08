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
#include "kernel/cpu/native/shift_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace shift_cpu {
int ShiftCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  size_t input_count = inputs.size();
  if (input_count != 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_count
                      << " input(s).";
  }

  size_t output_count = outputs.size();
  if (output_count != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_count
                      << " output(s).";
  }

  auto input_shape = inputs[kInputIndex]->GetShapeVector();

  periods_ = GetValue<int64_t>(primitive_->GetAttr(PERIODS));
  auto axis = GetValue<int64_t>(primitive_->GetAttr(AXIS));
  size_t axis_t = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  if (axis_t >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be less than the dimension of input tensor "
                      << input_shape.size() << "D, but got " << axis_t;
  }

  axisIterator_.Init(input_shape, axis_t);

  // index calculation
  if (periods_ > 0) {
    fill_begin_ = 0;
    fill_size_ = periods_;

    copy_src_begin_ = 0;
    copy_dst_begin_ = periods_;
    copy_size_ = SizeToLong(input_shape[axis]) - periods_;
  } else if (periods_ < 0) {
    fill_begin_ = SizeToLong(input_shape[axis]) + periods_;
    fill_size_ = -periods_;

    copy_src_begin_ = -periods_;
    copy_dst_begin_ = 0;
    copy_size_ = SizeToLong(input_shape[axis]) + periods_;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Shift does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return KRET_OK;
}

template <typename T>
bool ShiftCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                     const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << inputs.size()
                      << " input(s).";
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << outputs.size()
                      << " output(s).";
  }
  auto input = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  const auto fill_value = reinterpret_cast<T *>(inputs[kIndex1]->device_ptr())[0];
  auto output = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());

  if (outputs[kIndex0]->size() != inputs[kIndex0]->size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory size of output must be equal to the memory size "
                         "of the first input, but got the memory size of output: "
                      << outputs[kIndex0]->size()
                      << " and the memory size of the first input: " << inputs[kIndex0]->size();
  }

  // if periods_ is 0, do nothing
  if (periods_ == 0) {
    // directly copy input to output
    auto ret = memcpy_s(output, outputs[kIndex0]->size(), input, inputs[kIndex0]->size());
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed";
    }
    return true;
  }

  const int64_t outer_size = SizeToLong(axisIterator_.OuterSize());
  const int64_t axis_size = SizeToLong(axisIterator_.AxisSize());
  const int64_t inner_size = SizeToLong(axisIterator_.InnerSize());

  // periods is larger than size, all value of the tensor would be fill_value
  if (std::abs(periods_) >= axis_size) {
    (void)std::fill_n(output, outer_size * axis_size * inner_size, fill_value);
    return true;
  }

  if (inputs[0]->size() != LongToSize(outer_size * axis_size * inner_size) * sizeof(T)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory size of inputs error.";
  }

  // check if the tensor is linear
  if ((inner_size == 1) && (outer_size == 1)) {
    // treat it as a simple 1D array
    size_t copy_size = copy_size_ * sizeof(T);
    size_t dst_max_size = outputs[kIndex0]->size() - copy_dst_begin_;
    auto ret = memcpy_s(output + copy_dst_begin_, dst_max_size, input + copy_src_begin_, copy_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed";
    }
    (void)std::fill_n(output + fill_begin_, fill_size_, fill_value);
    return true;
  }

  // normal procedure
  auto task = [this, fill_value, axis_size, inner_size, input, output, outputs](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t offset = i * LongToSize(axis_size) * LongToSize(inner_size);
      size_t input_offset = offset + LongToSize(copy_src_begin_ * inner_size);
      size_t output_offset = offset + LongToSize(copy_dst_begin_ * inner_size);
      size_t copy_size = copy_size_ * inner_size * sizeof(T);
      size_t dst_max_size = outputs[0]->size() - output_offset;
      auto ret = memcpy_s(output + output_offset, dst_max_size, input + input_offset, copy_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed, ret=" << ret;
      }
      size_t fill_offset = offset + LongToSize(fill_begin_ * inner_size);
      (void)std::fill_n(output + fill_offset, fill_size_ * inner_size, fill_value);
    }
  };
  ParallelLaunchAutoSearch(task, axisIterator_.OuterSize(), this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, ShiftCpuKernelMod::ShiftFunc>> ShiftCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &ShiftCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ShiftCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ShiftCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &ShiftCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ShiftCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> ShiftCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ShiftFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Shift, ShiftCpuKernelMod);
}  // namespace shift_cpu
}  // namespace kernel
}  // namespace mindspore
