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

#include "plugin/device/gpu/kernel/map_tensor/map_tensor_get_data_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, MapTensorGetDataGpuKernelMod::MapTensorGetDataLaunchFunc>>
  MapTensorGetDataGpuKernelMod::map_tensor_get_data_func_list_ = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeMapTensorType)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat),
     &MapTensorGetDataGpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeMapTensorType)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat),
     &MapTensorGetDataGpuKernelMod::LaunchKernel<int64_t, float>}};

std::vector<KernelAttr> MapTensorGetDataGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(map_tensor_get_data_func_list_.begin(), map_tensor_get_data_func_list_.end(),
                       std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MapTensorGetDataGpuKernelMod::MapTensorGetDataLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

bool MapTensorGetDataGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  MS_EXCEPTION_IF_NULL(base_operator->GetPrim());
  kernel_name_ = base_operator->GetPrim()->name();
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetDataInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetDataOutputNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = map_tensor_get_data_func_list_[index].second;
  InitSize(base_operator, inputs, outputs);
  return true;
}

int MapTensorGetDataGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  InitSize(base_operator, inputs, outputs);
  return KRET_OK;
}

template <typename KeyType, typename ValueType>
bool MapTensorGetDataGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetDataInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetDataOutputNum, kernel_name_);

  // The real hash table should be accessed by user data.
  if (input_user_data_.empty()) {
    MS_LOG(EXCEPTION) << "The hash table user data is not set yet.";
  }

  auto user_data = input_user_data_[kIndex0];
  MS_EXCEPTION_IF_NULL(user_data);
  auto hash_table_ptr = user_data->get<GPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(hash_table_ptr);
  return hash_table_ptr->GetKeysAndValues(static_cast<KeyType *>(outputs[kIndex0]->addr),
                                          static_cast<ValueType *>(outputs[kIndex1]->addr), stream_ptr);
}

bool MapTensorGetDataGpuKernelMod::InitSize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  // Return size 1 as the first input size for MapTensorGetData. Real map tensor is assigned by framework.
  input_size_list_.push_back(kSizeOne);

  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  auto key_size = outputs[kIndex0]->GetSizeInBytes();
  output_size_list_.push_back(key_size);

  MS_EXCEPTION_IF_NULL(outputs[kIndex1]);
  auto value_size = outputs[kIndex1]->GetSizeInBytes();
  output_size_list_.push_back(value_size);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MapTensorGetData, MapTensorGetDataGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
