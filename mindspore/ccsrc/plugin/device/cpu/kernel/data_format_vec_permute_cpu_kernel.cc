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

#include "plugin/device/cpu/kernel/data_format_vec_permute_cpu_kernel.h"
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDataFormatVecPermuteInputsNum = 1;
constexpr size_t kDataFormatVecPermuteOutputsNum = 1;
}  // namespace

void DataFormatVecPermuteCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  src_format_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "src_format");
  dst_format_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "dst_format");
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  input_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  output_type_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  dim_ = input_shape_.size();
  // check attr
  std::vector<size_t> shape1 = {4};
  std::vector<size_t> shape2 = {4, 2};
  if (src_format_ != "NHWC" && src_format_ != "NCHW") {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", src_format must be 'NHWC' or 'NCHW' , but got " << src_format_
                      << ".";
  }
  if (dst_format_ != "NHWC" && dst_format_ != "NCHW") {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", dst_format must be 'NHWC' or 'NCHW' , but got " << dst_format_
                      << ".";
  }
  if (input_shape_ == shape1) {
    if (output_shape_ != shape1) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", output must have the same shape as input, but got "
                        << output_shape_ << " .";
    }
  } else if (input_shape_ == shape2) {
    if (output_shape_ != shape2) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", output must have the same shape as input, but got "
                        << output_shape_ << " .";
    }
  } else {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", input shape must be (4, ) or (4, 2), but got " << input_shape_
                      << " .";
  }
  // check input and output type
  if (input_type_ != output_type_) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", input[" << input_type_ << "] and output[" << output_type_
                      << "] must have the same DataType.";
  }
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "DataFormatVecPermute does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool DataFormatVecPermuteCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDataFormatVecPermuteInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDataFormatVecPermuteOutputsNum, kernel_name_);
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t dim1 = 1;
  size_t dim2 = 2;
  if (dim_ == dim1) {
    for (size_t i = 0; i < dst_format_.size(); i++) {
      for (size_t j = 0; j < src_format_.size(); j++) {
        if (dst_format_[i] == src_format_[j]) {
          output[i] = input[j];
          break;
        }
      }
    }
  } else if (dim_ == dim2) {
    for (size_t i = 0; i < dst_format_.size(); i++) {
      for (size_t j = 0; j < src_format_.size(); j++) {
        if (dst_format_[i] == src_format_[j]) {
          output[i * dim2] = input[j * dim2];
          output[i * dim2 + 1] = input[j * dim2 + 1];
          break;
        }
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, DataFormatVecPermuteCpuKernelMod::DataFormatVecPermuteFunc>>
  DataFormatVecPermuteCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &DataFormatVecPermuteCpuKernelMod::LaunchKernel<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &DataFormatVecPermuteCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> DataFormatVecPermuteCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DataFormatVecPermuteFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DataFormatVecPermute, DataFormatVecPermuteCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
