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
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"

#include "plugin/device/cpu/hal/device/cpu_hash_table.h"
#include "plugin/device/cpu/kernel/map_tensor/map_tensor_put_cpu_kernel.h"
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, MapTensorPutCpuKernelMod::MapTensorPutLaunchFunc>>
  MapTensorPutCpuKernelMod::map_tensor_put_func_list_ = {{KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt32)
                                                            .AddInputAttr(kNumberTypeFloat32)
                                                            .AddOutputAttr(kObjectTypeMapTensorType),
                                                          &MapTensorPutCpuKernelMod::LaunchKernel<int32_t, float>},
                                                         {KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt64)
                                                            .AddInputAttr(kNumberTypeFloat32)
                                                            .AddOutputAttr(kObjectTypeMapTensorType),
                                                          &MapTensorPutCpuKernelMod::LaunchKernel<int64_t, float>}};

std::vector<KernelAttr> MapTensorPutCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    map_tensor_put_func_list_.begin(), map_tensor_put_func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MapTensorPutCpuKernelMod::MapTensorPutLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool MapTensorPutCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  MS_EXCEPTION_IF_NULL(base_operator->GetPrim());
  kernel_name_ = base_operator->GetPrim()->name();
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorPutInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorPutOutputNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = map_tensor_put_func_list_[index].second;
  input_key_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  input_value_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);

  if (base_operator->HasAttr(kAttrEnableEmbeddingStorage)) {
    enable_embedding_storage_ = GetValue<bool>(base_operator->GetAttr(kAttrEnableEmbeddingStorage));
  }
  if (base_operator->HasAttr(kAttrParameterKey)) {
    parameter_key_ = GetValue<int32_t>(base_operator->GetAttr(kAttrParameterKey));
  }

  return true;
}

int MapTensorPutCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs,
                                     const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();

  MS_EXCEPTION_IF_NULL(inputs.at(kIndex1));
  const auto &keys_shape = inputs.at(kIndex1)->GetShapeVector();
  MS_EXCEPTION_IF_NULL(inputs.at(kIndex2));
  const auto &values_shape = inputs.at(kIndex2)->GetShapeVector();
  if (IsDynamic(keys_shape) || IsDynamic(values_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }

  InitSizeLists(keys_shape, values_shape);
  return KRET_OK;
}

template <typename KeyType, typename ValueType>
bool MapTensorPutCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs) {
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorPutInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorPutOutputNum, kernel_name_);

  // The real hash table should be accessed by user data.
  if (input_user_data_.empty()) {
    MS_LOG(EXCEPTION) << "The hash table user data is not set yet.";
  }

  if (enable_embedding_storage_) {
    auto embedding_storage = embedding_storage_manager.Get(parameter_key_);
    MS_ERROR_IF_NULL(embedding_storage);
    if (!embedding_storage->Put({inputs[kIndex1]->addr, inputs[kIndex1]->size},
                                {inputs[kIndex2]->addr, inputs[kIndex2]->size})) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', Update sparse embedding storage failed, parameter key: " << parameter_key_;
      return false;
    }
    return true;
  }

  auto user_data = input_user_data_[kIndex0];
  MS_EXCEPTION_IF_NULL(user_data);
  auto hash_table_ptr = user_data->get<device::cpu::CPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(hash_table_ptr);
  return hash_table_ptr->Insert(static_cast<KeyType *>(inputs.at(kIndex1)->addr),
                                inputs.at(kIndex1)->size / sizeof(KeyType),
                                static_cast<ValueType *>(inputs.at(kIndex2)->addr), nullptr);
}

void MapTensorPutCpuKernelMod::InitSizeLists(const ShapeVector &keys_shape, const ShapeVector &values_shape) {
  // Return size 1 as the first input size and the output size for MapTensorPut. Real map tensor is assigned by
  // framework.
  input_size_list_.push_back(kSizeOne);
  output_size_list_.push_back(kSizeOne);

  auto keys_size = std::accumulate(keys_shape.begin(), keys_shape.end(), 1, std::multiplies{});
  MS_EXCEPTION_IF_ZERO("keys size", keys_size);
  input_size_list_.push_back(keys_size * input_key_type_size_);

  auto values_size = std::accumulate(values_shape.begin(), values_shape.end(), 1, std::multiplies{});
  MS_EXCEPTION_IF_ZERO("values size", values_size);
  input_size_list_.push_back(values_size * input_value_type_size_);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MapTensorPut, MapTensorPutCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
