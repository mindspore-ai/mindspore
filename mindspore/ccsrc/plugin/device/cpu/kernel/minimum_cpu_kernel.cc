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

#include "plugin/device/cpu/kernel/minimum_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMinimumInputsNum = 2;
constexpr size_t kMinimumOutputsNum = 1;

constexpr size_t kIdx2 = 2;
constexpr size_t kIdx3 = 3;
constexpr size_t kIdx4 = 4;
constexpr size_t kIdx5 = 5;
constexpr size_t kIdx6 = 6;
}  // namespace

void MinimumCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_y_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  TypeId input_x_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  TypeId input_y_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  size_t max_input_shape_size =
    input_x_shape_.size() > input_y_shape_.size() ? input_x_shape_.size() : input_y_shape_.size();
  for (size_t i = 0; i < output_shape_.size(); i++) {
    output_num_ *= static_cast<size_t>(output_shape_[i]);
  }
  if ((input_x_shape_.size() == 0 && input_y_shape_.size() != 0) ||
      (input_x_shape_.size() != 0 && input_y_shape_.size() == 0)) {
    InitInputTensorAndScalar(max_input_shape_size);
  } else if (max_input_shape_size == output_shape_.size() && output_shape_.size() != 0) {
    InitInputTensors(input_x_dtype, input_y_dtype);
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MinimumLaunchFunc> &pair) { return pair.first; });
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Minimum does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

void MinimumCpuKernelMod::InitInputTensorAndScalar(size_t max_input_shape_size) {
  if (max_input_shape_size != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output tensor must be equal to the max "
                         "dimension of inputs, but got the dimension of output tensor: "
                      << output_shape_.size() << " and the max dimension of inputs: " << max_input_shape_size;
  }
  need_broadcast_ = false;
}

void MinimumCpuKernelMod::InitInputTensors(TypeId input_x_dtype, TypeId input_y_dtype) {
  if (input_x_dtype == kNumberTypeBool && input_y_dtype == kNumberTypeBool) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input tensor types can not be both bool.";
  }
  // Check if the shape needs to be broadcast
  need_broadcast_ = IsBroadcast();
  if (need_broadcast_) {
    InitTensorBroadcastShape();
  }
}

template <typename T>
bool MinimumCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMinimumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMinimumOutputsNum, kernel_name_);
  T *input_x_ = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_y_ = reinterpret_cast<T *>(inputs[1]->addr);
  T *output_ = reinterpret_cast<T *>(outputs[0]->addr);
  BroadcastArith(input_x_, input_y_, output_);
  return true;
}

template <typename T>
void MinimumCpuKernelMod::BroadcastArith(const T *input_x, const T *input_y, T *output) const {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(output);
  if (need_broadcast_) {
    BroadcastArithKernel(broadcast_input_x_shape_[0], broadcast_input_x_shape_[1], broadcast_input_x_shape_[kIdx2],
                         broadcast_input_x_shape_[kIdx3], broadcast_input_x_shape_[kIdx4],
                         broadcast_input_x_shape_[kIdx5], broadcast_input_x_shape_[kIdx6], broadcast_input_y_shape_[0],
                         broadcast_input_y_shape_[1], broadcast_input_y_shape_[kIdx2], broadcast_input_y_shape_[kIdx3],
                         broadcast_input_y_shape_[kIdx4], broadcast_input_y_shape_[kIdx5],
                         broadcast_input_y_shape_[kIdx6], broadcast_output_shape_[0], broadcast_output_shape_[1],
                         broadcast_output_shape_[kIdx2], broadcast_output_shape_[kIdx3], broadcast_output_shape_[kIdx4],
                         broadcast_output_shape_[kIdx5], broadcast_output_shape_[kIdx6], input_x, input_y, output);
  } else {
    if (input_x_shape_.size() == 0 || input_y_shape_.size() == 0) {
      BroadcastArithOneScalarOneTensor(input_x, input_y, output);
    } else {
      BroadcastArithTensors(input_x, input_y, output);
    }
  }
}

bool MinimumCpuKernelMod::IsBroadcast() const {
  if (input_x_shape_.size() != input_y_shape_.size()) {
    return true;
  }
  for (size_t i = 0; i < input_x_shape_.size(); i++) {
    if (input_x_shape_[i] != input_y_shape_[i]) {
      return true;
    }
  }
  return false;
}

void MinimumCpuKernelMod::InitTensorBroadcastShape() {
  if (output_shape_.size() > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output must be less than or equal to 7, but got " << output_shape_.size();
  }
  broadcast_input_x_shape_.resize(max_dims_, 1);
  broadcast_input_y_shape_.resize(max_dims_, 1);
  broadcast_output_shape_.resize(max_dims_, 1);
  for (size_t i = 0; i < output_shape_.size(); i++) {
    broadcast_output_shape_[i] = output_shape_[i];
  }
  size_t input_x_dim_offset = output_shape_.size() - input_x_shape_.size();
  for (size_t j = 0; j < input_x_shape_.size(); j++) {
    broadcast_input_x_shape_[j + input_x_dim_offset] = input_x_shape_[j];
    input_x_num_ *= static_cast<size_t>(input_x_shape_[j]);
  }
  size_t input_y_dim_offset = output_shape_.size() - input_y_shape_.size();
  for (size_t k = 0; k < input_y_shape_.size(); k++) {
    if (need_broadcast_) {
      broadcast_input_y_shape_[k + input_y_dim_offset] = input_y_shape_[k];
      input_y_num_ *= static_cast<size_t>(input_y_shape_[k]);
    }
  }
}

// Broadcast comparison
int64_t MinimumCpuKernelMod::Index(const int64_t &index, const int64_t &dim) const { return dim == 1 ? 0 : index; }

// Broadcast Arithmetic
template <typename T>
void MinimumCpuKernelMod::BroadcastArithKernel(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3,
                                               const int64_t l4, const int64_t l5, const int64_t l6, const int64_t r0,
                                               const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                                               const int64_t r5, const int64_t r6, const int64_t d0, const int64_t d1,
                                               const int64_t d2, const int64_t d3, const int64_t d4, const int64_t d5,
                                               const int64_t d6, const T *input_x, const T *input_y, T *output) const {
  for (size_t pos = 0; pos < output_num_; pos++) {
    auto pos_signed = SizeToLong(pos);
    int64_t i = pos_signed / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    int64_t j = pos_signed / (d2 * d3 * d4 * d5 * d6) % d1;
    int64_t k = pos_signed / (d3 * d4 * d5 * d6) % d2;
    int64_t l = pos_signed / (d4 * d5 * d6) % d3;
    int64_t m = pos_signed / (d5 * d6) % d4;
    int64_t n = pos_signed / d6 % d5;
    int64_t o = pos_signed % d6;

    int64_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] = MinimumFunc(input_x[LongToSize(l_index)], input_y[LongToSize(r_index)]);
  }
}

template <typename T>
void MinimumCpuKernelMod::BroadcastArithOneScalarOneTensor(const T *input_x, const T *input_y, T *output) const {
  if (input_x_shape_.size() == 0) {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = MinimumFunc(input_x[0], input_y[i]);
    }
  } else {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = MinimumFunc(input_x[i], input_y[0]);
    }
  }
}

template <typename T>
void MinimumCpuKernelMod::BroadcastArithTensors(const T *input_x, const T *input_y, T *output) const {
  for (size_t i = 0; i < output_num_; ++i) {
    output[i] = MinimumFunc(input_x[i], input_y[i]);
  }
}

std::vector<std::pair<KernelAttr, MinimumCpuKernelMod::MinimumLaunchFunc>> MinimumCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &MinimumCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &MinimumCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &MinimumCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &MinimumCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &MinimumCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &MinimumCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &MinimumCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &MinimumCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &MinimumCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &MinimumCpuKernelMod::LaunchKernel<double>}};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Minimum, MinimumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
