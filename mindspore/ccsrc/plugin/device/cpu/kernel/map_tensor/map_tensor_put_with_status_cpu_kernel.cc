/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <functional>
#include <utility>
#include <string>
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"

#include "plugin/device/cpu/hal/device/cpu_hash_table.h"
#include "plugin/device/cpu/kernel/map_tensor/map_tensor_put_with_status_cpu_kernel.h"
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, MapTensorPutWithStatusCpuKernelMod::MapTensorPutLaunchFunc>>
  MapTensorPutWithStatusCpuKernelMod::map_tensor_put_with_status_func_list_ = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeMapTensorType)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeMapTensorType),
     &MapTensorPutWithStatusCpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeMapTensorType)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kObjectTypeMapTensorType),
     &MapTensorPutWithStatusCpuKernelMod::LaunchKernel<int64_t, float>}};

std::vector<KernelAttr> MapTensorPutWithStatusCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    map_tensor_put_with_status_func_list_.begin(), map_tensor_put_with_status_func_list_.end(),
    std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MapTensorPutWithStatusCpuKernelMod::MapTensorPutLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

bool MapTensorPutWithStatusCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorPutWithStatusInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorPutWithStatusOutputNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = map_tensor_put_with_status_func_list_[index].second;
  input_key_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  input_value_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);

  return true;
}

int MapTensorPutWithStatusCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  ResetResource();

  MS_EXCEPTION_IF_NULL(inputs.at(kIndex1));
  const auto &keys_shape = inputs.at(kIndex1)->GetShapeVector();
  MS_EXCEPTION_IF_NULL(inputs.at(kIndex2));
  const auto &values_shape = inputs.at(kIndex2)->GetShapeVector();
  MS_EXCEPTION_IF_NULL(inputs.at(kIndex3));
  if (IsDynamic(keys_shape) || IsDynamic(values_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }

  InitSizeLists(keys_shape, values_shape);
  return KRET_OK;
}

template <typename KeyType, typename ValueType>
bool MapTensorPutWithStatusCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &,
                                                      const std::vector<KernelTensor *> &outputs) {
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorPutWithStatusInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorPutWithStatusOutputNum, kernel_name_);

  // The real hash table should be accessed by user data.
  if (input_user_data_.empty()) {
    MS_LOG(EXCEPTION) << "The hash table user data is not set yet.";
  }

  auto user_data = input_user_data_[kIndex0];
  MS_EXCEPTION_IF_NULL(user_data);
  auto hash_table_ptr = user_data->get<device::cpu::CPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(hash_table_ptr);
  return hash_table_ptr->Insert(static_cast<KeyType *>(inputs.at(kIndex1)->device_ptr()),
                                inputs.at(kIndex1)->size() / sizeof(KeyType),
                                static_cast<ValueType *>(inputs.at(kIndex2)->device_ptr()),
                                static_cast<HashTableElementStatus *>(inputs.at(kIndex3)->device_ptr()), nullptr);
}

void MapTensorPutWithStatusCpuKernelMod::InitSizeLists(const ShapeVector &keys_shape, const ShapeVector &values_shape) {
  // Return size 1 as the first input size and the output size for MapTensorPutWithStatus. Real map tensor is assigned
  // by framework.
  output_size_list_.push_back(kSizeOne);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MapTensorPutWithStatus, MapTensorPutWithStatusCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
