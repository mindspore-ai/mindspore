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
#include <memory>
#include <map>
#include "mindspore/core/ops/data_format_vec_permute.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDataFormatVecPermuteInputsNum = 1;
constexpr size_t kDataFormatVecPermuteOutputsNum = 1;
}  // namespace

bool DataFormatVecPermuteCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "DataFormatVecPermute does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;

  auto kernel_ptr = std::make_shared<ops::DataFormatVecPermute>(base_operator->GetPrim());
  src_format_ = kernel_ptr->get_src_format();
  dst_format_ = kernel_ptr->get_dst_format();
  input_type_ = inputs[0]->GetDtype();
  output_type_ = outputs[0]->GetDtype();
  return true;
}

int DataFormatVecPermuteCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[0]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[0]->GetDeviceShapeAdaptively();
  dim_ = input_shape_.size();
  return KRET_OK;
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
