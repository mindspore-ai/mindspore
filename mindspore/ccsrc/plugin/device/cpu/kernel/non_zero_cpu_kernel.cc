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

#include "mindspore/ccsrc/plugin/device/cpu/kernel/non_zero_cpu_kernel.h"
#include <algorithm>
#include <typeinfo>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
constexpr size_t kInputMinDim = 1;
constexpr size_t kOutputDim = 2;
}  // namespace

void NonZeroCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (AnfAlgo::IsShapesDynamic({input_shape})) {
    return;
  }
  input_shape_ = Convert2SizeT(input_shape);
  output_shape_ = Convert2SizeTClipNeg(output_shape);
  input_rank_ = input_shape_.size();
  node_wpt_ = kernel_node;
  if (input_shape_.size() < kInputMinDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should be greater than or equal to "
                      << kInputMinDim << ", but got " << input_shape_.size() << ".";
  }

  if (output_shape_.size() != kOutputDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output should be equal to " << kOutputDim
                      << ", but got " << output_shape_.size() << ".";
  }

  if (output_shape_[1] != input_rank_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the shape[1] of output should be equal to rank of input"
                      << ", but got output_shape[1]=" << output_shape_[1] << " and x_rank=" << input_rank_ << ".";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NonZeroFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "NonZero does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool NonZeroCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto input_addr = static_cast<T *>(inputs[0]->addr);
  auto output_addr = static_cast<int64_t *>(outputs[0]->addr);

  size_t input_num = inputs[0]->size / sizeof(T);
  size_t non_zero_num = NonZeroCompute(input_addr, output_addr, input_num);
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node_ << ".";
  }
  ShapeVector output_shape = {SizeToLong(non_zero_num), SizeToLong(input_rank_)};
  std::vector<TypeId> dtype = {AnfAlgo::GetOutputDeviceDataType(node_, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtype, {output_shape}, node_.get());

  return true;
}

template <typename T>
size_t NonZeroCpuKernelMod::NonZeroCompute(const T *input, int64_t *output, size_t input_num) {
  size_t non_zero_count = 0;
  std::vector<size_t> dim_strides(input_rank_, 1);

  for (size_t i = input_rank_ - 1; i >= 1; --i) {
    dim_strides[i - 1] = dim_strides[i] * input_shape_[i];
  }

  for (size_t elem_i = 0; elem_i < input_num; ++elem_i) {
    auto zero = static_cast<T>(0);
    if constexpr (std::is_same_v<T, double>) {
      if (common::IsDoubleEqual(input[elem_i], zero)) {
        continue;
      }
    } else {
      if constexpr (std::is_same_v<T, float>) {
        if (common::IsFloatEqual(input[elem_i], zero)) {
          continue;
        }
      } else {
        if (input[elem_i] == zero) {
          continue;
        }
      }
    }
    size_t index = elem_i;
    for (size_t pos_j = 0; pos_j < input_rank_; ++pos_j) {
      output[non_zero_count * input_rank_ + pos_j] = static_cast<int64_t>(index / dim_strides[pos_j]);
      index %= dim_strides[pos_j];
    }
    non_zero_count++;
  }
  return non_zero_count;
}

std::vector<std::pair<KernelAttr, NonZeroCpuKernelMod::NonZeroFunc>> NonZeroCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64),
   &NonZeroCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> NonZeroCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NonZeroFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NonZero, NonZeroCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
