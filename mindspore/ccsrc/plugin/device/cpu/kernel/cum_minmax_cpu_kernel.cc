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
#include "mindspore/core/ops/cummin.h"
#include "mindspore/core/ops/cummax.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kCumInputsNum = 1;
constexpr size_t kCumOutputsNum = 2;
constexpr size_t kMinSizeUsingMT = 1000;

template <typename T>
inline bool IsNan(T x) {
  return std::isnan(x);
}

template <>
inline bool IsNan<float16>(float16 x) {
  return isnan(x);
}

#ifdef _MSC_VER
template <>
inline bool IsNan<int8_t>(int8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint8_t>(uint8_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int16_t>(int16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint16_t>(uint16_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int32_t>(int32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint32_t>(uint32_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<int64_t>(int64_t x) {
  return isnan(static_cast<double>(x));
}

template <>
inline bool IsNan<uint64_t>(uint64_t x) {
  return isnan(static_cast<double>(x));
}
#endif

template <typename T, typename S, typename BinaryOp>
inline void CumMinMax(const T *input_ptr, T *value_ptr, S *index_ptr, BinaryOp op, size_t axis_inner_size,
                      size_t axis_size, size_t inner_size, size_t start, size_t end) {
  size_t outer_idx = (start / inner_size) * axis_inner_size;
  size_t inner_idx = start % inner_size;
  for (size_t i = start; i < end; i++) {
    size_t offset = (outer_idx + inner_idx);
    auto cur_input_ptr = input_ptr + offset;
    auto cur_value_ptr = value_ptr + offset;
    auto cur_index_ptr = index_ptr + offset;
    T out_val = *cur_value_ptr = *cur_input_ptr;
    S out_idx = *cur_index_ptr = 0;
    for (size_t j = 1; j < axis_size; j++) {
      cur_input_ptr += inner_size;
      cur_value_ptr += inner_size;
      cur_index_ptr += inner_size;
      T cur_val = *cur_input_ptr;
      if (IsNan(cur_val) || (!IsNan(out_val) && op(cur_val, out_val))) {
        out_val = cur_val;
        out_idx = static_cast<S>(j);
      }
      *cur_value_ptr = out_val;
      *cur_index_ptr = out_idx;
    }
    if (++inner_idx == inner_size) {
      inner_idx = 0;
      outer_idx += axis_inner_size;
    }
  }
}
}  // namespace

bool CumMinMaxCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCumOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  base_operator_ = base_operator;
  kernel_func_ = func_list_[cum_op_type_][index].second;
  switch (cum_op_type_) {
    case CUMMIN: {
      auto kernel_ptr = std::make_shared<ops::Cummin>(base_operator->GetPrim());
      axis_ = kernel_ptr->get_axis();
      break;
    }
    case CUMMAX: {
      auto kernel_ptr = std::make_shared<ops::Cummax>(base_operator->GetPrim());
      axis_ = kernel_ptr->get_axis();
      break;
    }
    default: {
      MS_LOG(ERROR) << "CumMin/CumMax Something unexpected happened!";
      return false;
    }
  }
  return true;
}

int CumMinMaxCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != 0) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto rank = SizeToLong(input_shape.size());
  auto axis_input = GetValue<int64_t>(base_operator_->GetAttr(AXIS));
  auto axis = axis_input < 0 ? LongToSize(axis_input + rank) : LongToSize(axis_input);
  outer_size_ = inner_size_ = axis_size_ = 1;
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (i < axis) {
      outer_size_ *= input_shape.at(i);
    } else if (i > axis) {
      inner_size_ *= input_shape.at(i);
    } else {
      axis_size_ = input_shape.at(i);
    }
  }
  return KRET_OK;
}

template <typename T, typename S>
bool CumMinMaxCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto element_size = (outer_size_ * inner_size_) * axis_size_;
  if (element_size == 0) {
    return true;
  }
  auto input_ptr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto value_ptr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto index_ptr = reinterpret_cast<S *>(outputs[kIndex1]->addr);
  auto any = [](auto... args) -> bool { return ((args == nullptr) || ...); };
  if (any(input_ptr, value_ptr, index_ptr)) {
    return false;
  }
  CTask task;
  switch (cum_op_type_) {
    case CUMMIN: {
      task =
        std::bind(CumMinMax<T, S, std::less_equal<T>>, input_ptr, value_ptr, index_ptr, std::less_equal<T>(),
                  (axis_size_ * inner_size_), axis_size_, inner_size_, std::placeholders::_1, std::placeholders::_2);
      break;
    }
    case CUMMAX: {
      task =
        std::bind(CumMinMax<T, S, std::greater_equal<T>>, input_ptr, value_ptr, index_ptr, std::greater_equal<T>(),
                  (axis_size_ * inner_size_), axis_size_, inner_size_, std::placeholders::_1, std::placeholders::_2);
      break;
    }
    default: {
      MS_LOG(ERROR) << "CumMin/CumMax Something unexpected happened!";
      return false;
    }
  }
  auto batch_size = outer_size_ * inner_size_;
  if (batch_size < kMinSizeUsingMT) {
    task(0, batch_size);
  } else {
    ParallelLaunchAutoSearch(task, batch_size, this, &parallel_search_info_, pool_);
  }
  return true;
}

// Note that in definition of primitive, Cummin return int32 as indices and Cummax return int64 as indices. (see
// cummax.cc and cummin.cc).
std::map<CumOpType, std::vector<std::pair<KernelAttr, CumMinMaxCpuKernelMod::CumMinMaxLaunchFunc>>>
  CumMinMaxCpuKernelMod::func_list_ = {
    {CUMMIN,
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
    {CUMMAX,
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
  auto iter = func_list_.find(cum_op_type_);
  if (iter == func_list_.end()) {
    MS_LOG(EXCEPTION) << "Cum_minmax cpu does not support " << cum_op_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(
    iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CumMinMaxCpuKernelMod::CumMinMaxLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Cummin,
                                 []() { return std::make_shared<CumMinMaxCpuKernelMod>(CUMMIN); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Cummax,
                                 []() { return std::make_shared<CumMinMaxCpuKernelMod>(CUMMAX); });
}  // namespace kernel
}  // namespace mindspore
