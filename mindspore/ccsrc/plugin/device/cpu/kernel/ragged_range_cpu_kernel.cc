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

bool RaggedRangeCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto is_match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  input_type_ = inputs[kIndex0]->GetDtype();
  tsplits_type_ = outputs[kIndex0]->GetDtype();
  return true;
}

int RaggedRangeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto starts_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  size_t starts_dim = starts_shape.size();
  auto limits_shape = inputs[kIndex1]->GetDeviceShapeAdaptively();
  size_t limits_dim = limits_shape.size();
  auto deltas_shape = inputs[kIndex2]->GetDeviceShapeAdaptively();
  size_t deltas_dim = deltas_shape.size();

  broadcast_starts_ = starts_dim == 0;
  broadcast_limits_ = limits_dim == 0;
  broadcast_deltas_ = deltas_dim == 0;
  if (!broadcast_starts_) in_sizes_.push_back(starts_shape[0]);
  if (!broadcast_limits_) in_sizes_.push_back(limits_shape[0]);
  if (!broadcast_deltas_) in_sizes_.push_back(deltas_shape[0]);
  return KRET_OK;
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
