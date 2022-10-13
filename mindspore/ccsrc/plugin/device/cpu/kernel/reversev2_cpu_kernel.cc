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

#include "plugin/device/cpu/kernel/reversev2_cpu_kernel.h"

#include <algorithm>
#include <utility>

#include "Eigen/Core"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kReverseV2InputsNum = 1;
constexpr size_t kReverseV2OutputsNum = 1;
constexpr int64_t kInputDim = 9;
}  // namespace

void ReverseV2CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (common::AnfAlgo::HasNodeAttr("axis", kernel_node)) {
    axis_shape_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "axis");
  }
  input_dims_ = SizeToLong(input_shape_.size());
  axis_dims_ = SizeToLong(axis_shape_.size());
  if (input_dims_ >= kInputDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should less than " << kInputDim
                      << ", but got " << input_dims_;
  }
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For ReverseV2, ReverseV2 type should be uint8_t, uint16_t, int8_t, int16_t, "
                         "int32_t, int64_t, float16, float, double, complex64, complex128, but got data type: "
                      << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool ReverseV2CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReverseV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReverseV2OutputsNum, kernel_name_);

  auto input_data = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_data = reinterpret_cast<T *>(outputs[0]->addr);
  int64_t shape_element = 1;
  for (int64_t i = 0; i < input_dims_; ++i) {
    shape_element *= SizeToLong(input_shape_[i]);
  }
  if (axis_dims_ == 0) {
    for (int64_t i = 0; i < shape_element; i++) {
      *(output_data + i) = *(input_data + i);
    }
    return true;
  }
  std::vector<bool> reverse_shape;
  for (int64_t i = 0; i < input_dims_; i++) {
    reverse_shape.push_back(false);
  }
  for (int64_t i = 0; i < axis_dims_; ++i) {
    int64_t realdim =
      SizeToLong(axis_shape_[i]) < 0 ? input_dims_ + SizeToLong(axis_shape_[i]) : SizeToLong(axis_shape_[i]);
    reverse_shape[realdim] = true;
  }
  int64_t front = 1;
  bool redo = false;
  for (int64_t j = 0; j < input_dims_; j++) {
    front = front * SizeToLong(input_shape_[j]);
    if ((j != input_dims_ - 1) && (SizeToLong(reverse_shape[j]))) {
      if (redo) {
        (void)memcpy(input_data, output_data, shape_element * sizeof(T));
      }
      int64_t row_size = shape_element / front;
      int64_t input_forward = (SizeToLong(input_shape_[j]) - 1) * row_size;
      int64_t save = input_forward;
      int64_t output_forward = 0;
      int64_t behind = shape_element / (front / SizeToLong(input_shape_[j]));
      for (int64_t k = 0; k < front / SizeToLong(input_shape_[j]); k++) {
        int64_t remain = SizeToLong(input_shape_[j]);
        while (remain > 0) {
          (void)memcpy(output_data + output_forward, input_data + input_forward, row_size * sizeof(T));
          input_forward = input_forward - row_size;
          output_forward = output_forward + row_size;
          remain--;
        }
        if (j != 0) {
          save = save + behind;
          input_forward = save;
        }
      }
      redo = true;
    } else if ((j == input_dims_ - 1) && (SizeToLong(reverse_shape[j]))) {
      if (redo) {
        (void)memcpy(input_data, output_data, shape_element * sizeof(T));
      }
      int64_t output_forward = 0;
      for (int64_t k = 0; k < shape_element / SizeToLong(input_shape_[j]); k++) {
        for (int64_t i = 0; i < SizeToLong(input_shape_[j]); i++) {
          *(output_data + output_forward) = *(input_data - 1 - i + (k + 1) * SizeToLong(input_shape_[j]));
          output_forward++;
        }
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, ReverseV2CpuKernelMod::ReverseV2Func>> ReverseV2CpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &ReverseV2CpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &ReverseV2CpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &ReverseV2CpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &ReverseV2CpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &ReverseV2CpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &ReverseV2CpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ReverseV2CpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ReverseV2CpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &ReverseV2CpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &ReverseV2CpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &ReverseV2CpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> ReverseV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReverseV2Func> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ReverseV2, ReverseV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
