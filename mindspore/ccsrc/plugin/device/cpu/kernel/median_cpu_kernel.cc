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

#include "plugin/device/cpu/kernel/median_cpu_kernel.h"

#include <algorithm>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMedianInputsNum = 1;
constexpr size_t kMedianOutputsNum = 2;
constexpr size_t kIndex0 = 0;
constexpr size_t kHalf = 2;
}  // namespace

void MedianCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  global_median_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "global_median");
  axis_ = static_cast<int>(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "axis"));
  keepdim_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "keep_dims");
  input_dim_ = input_shape_.size();
  input_num_elements_ = 1;

  if (input_dim_ != 0) {
    if (axis_ > static_cast<int>(input_dim_ - 1) || axis_ < static_cast<int>(-input_dim_)) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the axis must be in [" << -input_dim_ << ","
                               << input_dim_ << "), but got " << axis_ << ".";
    }
    for (size_t i = 0; i < input_dim_; i++) {
      input_num_elements_ *= input_shape_[i];
    }
  } else {
    if (axis_ > 0 || axis_ < -1) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the axis must be in [" << -1 << "," << 1
                               << "), but got " << axis_ << ".";
    }
  }
}

bool MedianCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMedianInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMedianOutputsNum, kernel_name_);
  if (global_median_ == false) {
    switch (input_type_) {
      case kNumberTypeInt16:
        return MedianCompute<int16_t>(inputs, outputs);
      case kNumberTypeInt32:
        return MedianCompute<int32_t>(inputs, outputs);
      case kNumberTypeInt64:
        return MedianCompute<int64_t>(inputs, outputs);
      case kNumberTypeFloat32:
        return MedianCompute<float>(inputs, outputs);
      case kNumberTypeFloat64:
        return MedianCompute<double>(inputs, outputs);
      default:
        MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                                << "', the input data type must be in int16, int32, int64, float32, double, but got "
                                << input_type_ << ".";
    }
  } else {
    switch (input_type_) {
      case kNumberTypeInt16:
        return GlobalMedianCompute<int16_t>(inputs, outputs);
      case kNumberTypeInt32:
        return GlobalMedianCompute<int32_t>(inputs, outputs);
      case kNumberTypeInt64:
        return GlobalMedianCompute<int64_t>(inputs, outputs);
      case kNumberTypeFloat32:
        return GlobalMedianCompute<float>(inputs, outputs);
      case kNumberTypeFloat64:
        return GlobalMedianCompute<double>(inputs, outputs);
      default:
        MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                                << "', the input data type must be in int16, int32, int64, float32, double, but got "
                                << input_type_ << ".";
    }
  }
  return true;
}

template <typename T>
bool MedianCpuKernelMod::GlobalMedianCompute(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &outputs) {
  auto *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output0 = reinterpret_cast<T *>(outputs[0]->addr);
  output_num_elements_ = 1;
  int64_t median_pos = static_cast<int64_t>((input_num_elements_ - 1) / kHalf);
  std::nth_element(input0, input0 + median_pos, input0 + static_cast<int64_t>(input_num_elements_));
  *output0 = *(input0 + median_pos);
  return true;
}

template <typename T>
bool MedianCpuKernelMod::MedianCompute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto *input0 = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output0 = reinterpret_cast<T *>(outputs[0]->addr);
  auto *output1 = reinterpret_cast<int64_t *>(outputs[1]->addr);
  if (input_dim_ == 0) {
    output_num_elements_ = 1;
    *output0 = *input0;
    *output1 = 0;
    return true;
  }
  if (axis_ < 0) {
    axis_ += input_dim_;
  }
  size_t dim_data_num = input_shape_[axis_];
  T *temp_median_vec = new T[dim_data_num];
  int64_t *temp_median_index_vec = new int64_t[dim_data_num];
  size_t group = 1;
  size_t jump = 1;
  int64_t median_pos = static_cast<int64_t>((dim_data_num - 1) / kHalf);
  if (axis_ != 0) {
    for (size_t i = 0; i < static_cast<size_t>(axis_); i++) {
      group *= input_shape_[i];
    }
  }
  if (axis_ != static_cast<int>(input_dim_ - 1)) {
    for (size_t i = static_cast<size_t>(axis_ + 1); i < input_dim_; i++) {
      jump *= input_shape_[i];
    }
  }
  auto start = input0;
  for (size_t i = 0; i < group; i++) {
    for (size_t j = 0; j < jump; j++) {
      for (size_t k = 0; k < dim_data_num; k++) {
        auto num_index = start + k * jump + j;
        temp_median_index_vec[k] = static_cast<int64_t>(k);
        temp_median_vec[k] = *num_index;
      }
      std::nth_element(
        temp_median_index_vec, temp_median_index_vec + median_pos, temp_median_index_vec + dim_data_num,
        [&temp_median_vec, dim_data_num](size_t pos1, size_t pos2) {
          return (*(temp_median_vec + pos1) < *(temp_median_vec + pos2)) ||
                 (pos1 < dim_data_num && *(temp_median_vec + pos1) == *(temp_median_vec + pos2) && pos1 < pos2);
        });
      std::nth_element(temp_median_vec, temp_median_vec + median_pos, temp_median_vec + dim_data_num);
      *(output0 + i * jump + j) = *(temp_median_vec + median_pos);
      *(output1 + i * jump + j) = *(temp_median_index_vec + median_pos);
    }
    if (i != group - 1) {
      start += jump * dim_data_num;
    }
  }
  delete[] temp_median_vec;
  delete[] temp_median_index_vec;
  return true;
}

std::vector<KernelAttr> MedianCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Median, MedianCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
