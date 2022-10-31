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

#include "plugin/device/gpu/kernel/map_tensor/map_tensor_get_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <string>
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, MapTensorGetGradGpuKernelMod::MapTensorGetGradLaunchFunc>>
  MapTensorGetGradGpuKernelMod::map_tensor_get_grad_func_list_ = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeMapTensorType)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kObjectTypeMapTensorType),
     &MapTensorGetGradGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeMapTensorType)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kObjectTypeMapTensorType),
     &MapTensorGetGradGpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> MapTensorGetGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(map_tensor_get_grad_func_list_.begin(), map_tensor_get_grad_func_list_.end(),
                       std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MapTensorGetGradGpuKernelMod::MapTensorGetGradLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

bool MapTensorGetGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  MS_EXCEPTION_IF_NULL(base_operator->GetPrim());
  kernel_name_ = base_operator->GetPrim()->name();
  // Check the inputs and outputs num.
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapTensorGetGradInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapTensorGetGradOutputNum, kernel_name_);

  // Check the kernel attr.
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  // Get kernel launch function.
  kernel_launch_func_ = map_tensor_get_grad_func_list_[index].second;
  InitSize(base_operator, inputs, outputs);

  // The output of this kernel is dynamic, so need update the output shape.
  is_need_retrieve_output_shape_ = true;
  outputs_ = outputs;
  return true;
}

int MapTensorGetGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  InitSize(base_operator, inputs, outputs);

  auto keys = inputs.at(kIndex1);
  MS_EXCEPTION_IF_NULL(keys);
  auto keys_shape = keys->GetShapeVector();
  for (size_t i = 0; i < keys_shape.size(); i++) {
    key_size_ *= keys_shape[i];
  }

  value_dims_.push_back(key_size_);
  auto dout = inputs.at(kIndex2);
  MS_EXCEPTION_IF_NULL(dout);
  auto dout_shape = dout->GetShapeVector();
  for (size_t i = keys_shape.size(); i < dout_shape.size(); i++) {
    value_dims_.push_back(dout_shape[i]);
  }

  return KRET_OK;
}

void MapTensorGetGradGpuKernelMod::SyncData() {
  MS_EXCEPTION_IF_CHECK_FAIL(outputs_.size() == 1, "The outputs number of kernel MapTensorGetGrad should be 1");
  outputs_[0]->SetShapeVector(value_dims_);
}

template <typename KeyType>
bool MapTensorGetGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &workspace,
                                                const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  // The real hash table should be accessed by user data.
  if (output_user_data_.empty()) {
    MS_LOG(EXCEPTION) << "The hash table user data is not set yet.";
  }

  auto user_data = output_user_data_[kIndex0];
  MS_EXCEPTION_IF_NULL(user_data);

  auto hash_table_value_type = user_data->get<TypeId>(kHashTableValueType);
  MS_EXCEPTION_IF_NULL(hash_table_value_type);
  TypeId value_type = *hash_table_value_type;
  if (value_type == kNumberTypeFloat32) {
    auto hash_table_ptr = user_data->get<GPUHashTable<KeyType, float>>(kUserDataData);
    MS_EXCEPTION_IF_NULL(hash_table_ptr);

    return hash_table_ptr->Insert(reinterpret_cast<KeyType *>(inputs[kIndex1]->addr),
                                  inputs[kIndex1]->size / sizeof(KeyType),
                                  reinterpret_cast<float *>(inputs[kIndex2]->addr), stream_ptr);
  } else {
    MS_LOG(EXCEPTION) << "GPU hash table does not support value type:" << value_type;
  }
  return false;
}

bool MapTensorGetGradGpuKernelMod::InitSize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  // Put size one for map tensor input, the real memory will allocate by gpu hash table dynamically.
  input_size_list_.push_back(kSizeOne);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  input_size_list_.push_back(inputs[kIndex1]->GetSizeInBytes());
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  input_size_list_.push_back(inputs[kIndex2]->GetSizeInBytes());

  // Put size one for map tensor output, the real memory will allocate by gpu hash table dynamically.
  output_size_list_.push_back(kSizeOne);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MapTensorGetGrad, MapTensorGetGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
