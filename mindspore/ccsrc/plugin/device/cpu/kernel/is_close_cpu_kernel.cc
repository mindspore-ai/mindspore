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

#include "plugin/device/cpu/kernel/is_close_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include <memory>
#include "mindspore/core/ops/is_close.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIsCloseInputsNum = 2;
constexpr size_t kIsCloseOutputsNum = 1;
template <typename T>
inline bool IsClose(T a, T b, float rtol, float atol, bool equal_nan) {
  if (std::equal_to<T>()(a, b)) {
    return true;
  }
  if (equal_nan && std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::equal_to<float>()(atol, 0) && std::equal_to<float>()(rtol, 0)) {
    return false;
  }
  auto left_side = std::abs(a - b);
  auto right_side = atol + (rtol * std::abs(b));
  return std::isfinite(left_side) && left_side <= right_side;
}
}  // namespace
bool IsCloseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsCloseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsCloseOutputsNum, kernel_name_);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::IsClose>(base_operator->GetPrim());
  rtol_ = kernel_ptr->get_rtol();
  atol_ = kernel_ptr->get_atol();
  equal_nan_ = kernel_ptr->get_equal_nan();
  return true;
}

int IsCloseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto other_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());
  auto output_shape = LongVecToSizeVec(outputs.at(kIndex0)->GetShapeVector());
  has_null_input_ = CheckNullInput(input_shape);
  has_null_input_ = has_null_input_ || CheckNullInput(other_shape);
  if (has_null_input_) {
    return KRET_OK;
  }
  is_need_broadcast_ = input_shape != other_shape;
  if (is_need_broadcast_) {
    GetBroadCastIndex(input_shape, output_shape, &index_list1_);
    GetBroadCastIndex(other_shape, output_shape, &index_list2_);
  }
  return KRET_OK;
}

template <typename T>
bool IsCloseCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsCloseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsCloseOutputsNum, kernel_name_);
  auto input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto other = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto output = reinterpret_cast<bool *>(outputs[kIndex0]->addr);

  if (has_null_input_) {
    return true;
  }

  CTask task;
  if (!is_need_broadcast_) {
    task = [this, &input, &other, &output](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
          auto a = static_cast<float>(input[i]);
          auto b = static_cast<float>(other[i]);
          output[i] = IsClose<float>(a, b, rtol_, atol_, equal_nan_);
        } else {
          output[i] = IsClose(input[i], other[i], rtol_, atol_, equal_nan_);
        }
      }
    };
  } else {
    task = [this, &input, &other, &output](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto idx1 = index_list1_[i];
        auto idx2 = index_list2_[i];
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
          auto a = static_cast<float>(input[idx1]);
          auto b = static_cast<float>(other[idx2]);
          output[i] = IsClose<float>(a, b, rtol_, atol_, equal_nan_);
        } else {
          output[i] = IsClose(input[idx1], other[idx2], rtol_, atol_, equal_nan_);
        }
      }
    };
  }
  size_t elem_num = outputs[kIndex0]->size / sizeof(bool);
  ParallelLaunch(task, elem_num, 0, this, pool_);
  return true;
}

const std::vector<std::pair<KernelAttr, IsCloseCpuKernelMod::KernelRunFunc>> &IsCloseCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, IsCloseCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
     &IsCloseCpuKernelMod::LaunchKernel<uint64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IsClose, IsCloseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
