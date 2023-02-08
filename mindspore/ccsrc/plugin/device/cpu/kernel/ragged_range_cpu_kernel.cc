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

#include "plugin/device/cpu/kernel/ragged_range_cpu_kernel.h"
#include <vector>
#include <cmath>
#include <type_traits>
#include <memory>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kOutputNum = 2;
const size_t kInputNum = 3;
const size_t kIndex0 = 0;
const size_t kIndex1 = 1;
const size_t kIndex2 = 2;
constexpr int64_t kParallelDataNums = 16 * 1024;
}  // namespace

void RaggedRangeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto starts_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto starts_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t starts_dim = starts_shape.size();
  auto limits_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  auto limits_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  size_t limits_dim = limits_shape.size();
  auto deltas_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
  auto deltas_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 2);
  size_t deltas_dim = deltas_shape.size();
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kInputNum, kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kOutputNum, kernel_node);
  if (starts_dim > 1) {
    MS_LOG(EXCEPTION) << "For RaggedRange, the dimension of RaggedRange input starts must be less than 2, but got "
                      << starts_dim << ".";
  }
  if (limits_dim > 1) {
    MS_LOG(EXCEPTION) << "For RaggedRange, the dimension of RaggedRange input limits must be less than 2, but got "
                      << limits_dim << ".";
  }
  if (deltas_dim > 1) {
    MS_LOG(EXCEPTION) << "For RaggedRange, the dimension of RaggedRange input deltas must be less than 2, but got "
                      << deltas_dim << ".";
  }
  if (!((starts_dim == limits_dim) && (starts_dim == deltas_dim) && (limits_dim == deltas_dim))) {
    MS_LOG(EXCEPTION) << "For RaggedRange, starts, limits, and deltas must have the same shape"
                      << ", but got starts (" << starts_dim << ",)"
                      << ", limits (" << limits_dim << ",)"
                      << ", deltas (" << deltas_dim << ",).";
  }

  broadcast_starts_ = starts_dim == 0;
  broadcast_limits_ = limits_dim == 0;
  broadcast_deltas_ = deltas_dim == 0;
  if (!broadcast_starts_) {
    in_sizes_.push_back(starts_shape[0]);
  }
  if (!broadcast_limits_) {
    in_sizes_.push_back(limits_shape[0]);
  }
  if (!broadcast_deltas_) {
    in_sizes_.push_back(deltas_shape[0]);
  }
  input_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  tsplits_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  if (starts_type != limits_type || starts_type != deltas_type || limits_type != deltas_type) {
    MS_LOG(EXCEPTION) << "For  RaggedRange, starts, limits, and deltas must have the same type, "
                      << "but got starts " << starts_type << ", limits " << limits_type << ", deltas " << deltas_type
                      << ".";
  }
}

bool RaggedRangeCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &workspace,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  size_t nrows = static_cast<size_t>(in_sizes_.empty() ? 1 : in_sizes_[0]);
  if (input_type_ == kNumberTypeInt32 && tsplits_type_ == kNumberTypeInt32) {
    RaggedRangeLaunch<int32_t, int32_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_,
                                        outputs);
  } else if (input_type_ == kNumberTypeInt32 && tsplits_type_ == kNumberTypeInt64) {
    RaggedRangeLaunch<int32_t, int64_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_,
                                        outputs);
  } else if (input_type_ == kNumberTypeInt64 && tsplits_type_ == kNumberTypeInt32) {
    RaggedRangeLaunch<int64_t, int32_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_,
                                        outputs);
  } else if (input_type_ == kNumberTypeInt64 && tsplits_type_ == kNumberTypeInt64) {
    RaggedRangeLaunch<int64_t, int64_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_,
                                        outputs);
  } else if (input_type_ == kNumberTypeFloat32 && tsplits_type_ == kNumberTypeInt32) {
    RaggedRangeLaunch<float, int32_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_, outputs);
  } else if (input_type_ == kNumberTypeFloat32 && tsplits_type_ == kNumberTypeInt64) {
    RaggedRangeLaunch<float, int64_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_, outputs);
  } else if (input_type_ == kNumberTypeFloat64 && tsplits_type_ == kNumberTypeInt32) {
    RaggedRangeLaunch<double, int32_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_, outputs);
  } else if (input_type_ == kNumberTypeFloat64 && tsplits_type_ == kNumberTypeInt64) {
    RaggedRangeLaunch<double, int64_t>(nrows, inputs, broadcast_starts_, broadcast_limits_, broadcast_deltas_, outputs);
  }
  return true;
}

template <typename T, typename TSPLITS>
void RaggedRangeCpuKernelMod::RaggedRangeLaunch(const size_t nrows, const std::vector<kernel::AddressPtr> &inputs,
                                                bool broadcast_starts, bool broadcast_limits, bool broadcast_deltas,
                                                const std::vector<kernel::AddressPtr> &outputs) const {
  T *starts_addr = static_cast<T *>(inputs[kIndex0]->addr);
  T *limits_addr = static_cast<T *>(inputs[kIndex1]->addr);
  T *deltas_addr = static_cast<T *>(inputs[kIndex2]->addr);
  TSPLITS *rt_nested_splits_addr = static_cast<TSPLITS *>(outputs[0]->addr);
  rt_nested_splits_addr[0] = 0;
  for (size_t row = 0; row < nrows; ++row) {
    T start = broadcast_starts ? starts_addr[0] : starts_addr[row];
    T limit = broadcast_limits ? limits_addr[0] : limits_addr[row];
    T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[row];
    if (delta == static_cast<T>(0)) {
      MS_LOG(EXCEPTION) << "For RaggedRange, requires delta != 0.";
    }
    rt_nested_splits_addr[row + 1] =
      rt_nested_splits_addr[row] + RaggedRangeCpuKernelMod::RangeSize<T, TSPLITS>(start, limit, delta);
  }
  T *rt_dense_values_addr = static_cast<T *>(outputs[1]->addr);
  if (nrows <= kParallelDataNums) {
    int value_index = 0;
    for (size_t row = 0; row < nrows; ++row) {
      TSPLITS row_size = rt_nested_splits_addr[row + 1] - rt_nested_splits_addr[row];
      T value = broadcast_starts ? starts_addr[0] : starts_addr[row];
      T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[row];
      for (TSPLITS i = 0; i < row_size; ++i) {
        rt_dense_values_addr[value_index++] = value;
        value += delta;
      }
    }
  } else {
    auto task = [this, &rt_dense_values_addr, &rt_nested_splits_addr, broadcast_starts, &starts_addr, broadcast_deltas,
                 &deltas_addr](size_t start, size_t end) {
      for (size_t row = start; row < end; row++) {
        TSPLITS row_size = rt_nested_splits_addr[row + 1] - rt_nested_splits_addr[row];
        T value = broadcast_starts ? starts_addr[0] : starts_addr[row];
        T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[row];
        TSPLITS y_offset = rt_nested_splits_addr[row];
        for (TSPLITS i = 0; i < row_size; ++i) {
          rt_dense_values_addr[y_offset++] = value;
          value += delta;
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, nrows);
  }
}

template <typename T, typename TSPLITS>
TSPLITS RaggedRangeCpuKernelMod::RangeSize(const T start, const T limit, const T delta) const {
  int64_t res;
  if (((delta > (T)0) && (limit < start)) || ((delta < (T)0) && (limit > start))) {
    res = 0;
  } else {
    res =
      (std::is_integral<T>::value
         ? ((std::abs(static_cast<int64_t>(limit) - static_cast<int64_t>(start)) +
             std::abs(static_cast<int64_t>(delta)) - 1) /
            std::abs(static_cast<int64_t>(delta)))
         : std::ceil(std::abs((static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta))));
  }
  return res;
}
std::vector<KernelAttr> RaggedRangeCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt32),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeInt32)
      .AddOutputAttr(kNumberTypeFloat64),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat64),
  };
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RaggedRange, RaggedRangeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
