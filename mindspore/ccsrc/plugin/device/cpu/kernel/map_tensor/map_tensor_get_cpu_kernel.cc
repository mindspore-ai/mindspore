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
#include "include/common/utils/utils.h"

#include "plugin/device/cpu/hal/device/cpu_hash_table.h"
#include "plugin/device/cpu/kernel/map_tensor/map_tensor_get_cpu_kernel.h"
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, MapTensorGetCpuKernelMod::MapTensorGetLaunchFunc>>
  MapTensorGetCpuKernelMod::map_tensor_get_func_list_ = {{KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt32)
                                                            .AddOutputAttr(kNumberTypeFloat32),
                                                          &MapTensorGetCpuKernelMod::LaunchKernel<int32_t, float>},
                                                         {KernelAttr()
                                                            .AddInputAttr(kObjectTypeMapTensorType)
                                                            .AddInputAttr(kNumberTypeInt64)
                                                            .AddOutputAttr(kNumberTypeFloat32),
                                                          &MapTensorGetCpuKernelMod::LaunchKernel<int64_t, float>}};

std::vector<KernelAttr> MapTensorGetCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    map_tensor_get_func_list_.begin(), map_tensor_get_func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MapTensorGetCpuKernelMod::MapTensorGetLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool MapTensorGetCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  insert_default_value_ = GetValue<bool>(primitive_->GetAttr(kAttrInsertDefaultValue));
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetOutputNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = map_tensor_get_func_list_[index].second;

  input_key_type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  output_type_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);

  if (primitive_->HasAttr(kAttrEnableEmbeddingStorage)) {
    enable_embedding_storage_ = GetValue<bool>(primitive_->GetAttr(kAttrEnableEmbeddingStorage));
  }
  if (primitive_->HasAttr(kAttrParameterKey)) {
    parameter_key_ = GetValue<int32_t>(primitive_->GetAttr(kAttrParameterKey));
  }

  return true;
}

int MapTensorGetCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  ResetResource();

  MS_EXCEPTION_IF_NULL(inputs.at(kIndex1));
  const auto &keys_shape = inputs.at(kIndex1)->GetShapeVector();
  MS_EXCEPTION_IF_NULL(outputs.at(kIndex0));
  const auto &output_shape = outputs.at(kIndex0)->GetShapeVector();

  if (IsDynamic(keys_shape) || IsDynamic(output_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }

  InitSizeLists(keys_shape, output_shape);
  return KRET_OK;
}

template <typename KeyType, typename ValueType>
bool MapTensorGetCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs) {
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetOutputNum, kernel_name_);

  // The real hash table should be accessed by user data.
  if (input_user_data_.empty()) {
    MS_LOG(EXCEPTION) << "The hash table user data is not set yet.";
  }

  if (enable_embedding_storage_) {
    auto embedding_storage = embedding_storage_manager.Get(parameter_key_);
    MS_ERROR_IF_NULL(embedding_storage);
    if (!embedding_storage->Get({inputs[kIndex1]->device_ptr(), inputs[kIndex1]->size()},
                                {outputs[kIndex0]->device_ptr(), outputs[kIndex0]->size()})) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', lookup embeddings from sparse embedding storage failed, parameter key: " << parameter_key_;
      return false;
    }
    return true;
  }

  auto user_data = input_user_data_[kIndex0];
  MS_EXCEPTION_IF_NULL(user_data);
  auto hash_table_ptr = user_data->get<device::cpu::CPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(hash_table_ptr);
  return hash_table_ptr->Find(static_cast<KeyType *>(inputs.at(kIndex1)->device_ptr()),
                              inputs.at(kIndex1)->size() / sizeof(KeyType), insert_default_value_,
                              static_cast<ValueType *>(outputs.at(kIndex0)->device_ptr()), nullptr);
}

void MapTensorGetCpuKernelMod::InitSizeLists(const ShapeVector &keys_shape, const ShapeVector &output_shape) {
  auto output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies{});
  MS_EXCEPTION_IF_ZERO("output size", output_size);
  output_size_list_.push_back(output_size * output_type_size_);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MapTensorGet, MapTensorGetCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
