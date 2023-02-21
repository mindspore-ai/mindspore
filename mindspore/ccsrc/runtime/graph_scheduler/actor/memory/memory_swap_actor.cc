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

#include "runtime/graph_scheduler/actor/memory/memory_swap_actor.h"

#include <map>

#include "runtime/graph_scheduler/device_tensor_store.h"

namespace mindspore {
namespace runtime {
void MemorySwapActor::FetchRealParameters(OpContext<mindspore::runtime::DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return;
  }
  for (auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    size_t input_index = IntToSize(input_data->index_);
    if (input_index >= real_parameters_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is out of range.");
    }
    real_parameters_[input_index] = input_data->data_;
  }
}

void MemorySwapActor::UpdateDeviceTensors(OpContext<mindspore::runtime::DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return;
  }
  const size_t total_device_tensor_num = fixed_device_tensor_num_ + data_iter->second.size();
  if (device_tensors_to_swap_.size() < total_device_tensor_num) {
    device_tensors_to_swap_.resize(total_device_tensor_num);
  }
  for (const auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    size_t input_index = IntToSize(input_data->index_);
    const size_t swap_device_tensor_index = input_index + fixed_device_tensor_num_;
    if (swap_device_tensor_index >= total_device_tensor_num) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is out of range.");
    }
    device_tensors_to_swap_[swap_device_tensor_index] = input_data->data_;
  }
}

std::vector<DeviceTensor *> MemorySwapActor::GetDeviceTensors(const std::vector<size_t> &indexes) {
  std::vector<DeviceTensor *> device_tensors;
  for (const auto index : indexes) {
    if (index >= device_tensors_to_swap_.size()) {
      MS_LOG(EXCEPTION) << "Device tensor index[" << index << "] out of range[" << device_tensors_to_swap_.size()
                        << "].";
    }
    device_tensors.emplace_back(device_tensors_to_swap_[index]);
  }
  return std::move(device_tensors);
}

void MemorySwapActor::AllocDeviceContinuousMem(const std::vector<DeviceTensor *> &device_tensors) {
  std::vector<size_t> size_list;
  for (const auto device_tensor : device_tensors) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    size_list.emplace_back(device_tensor->GetSize());
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]->device_res_manager_);
  const auto &device_ptrs = device_contexts_[0]->device_res_manager_->AllocateContinuousMemory(size_list);
  for (size_t i = 0; i < device_tensors.size(); ++i) {
    MS_EXCEPTION_IF_NULL(device_tensors[i]);
    device_tensors[i]->set_ptr(device_ptrs[i]);
    device_tensors[i]->set_from_mem_pool(true);
  }
}

void MemorySwapActor::Swap(device::StorageType to, const std::vector<DeviceTensor *> &device_tensors) {
  for (const auto &device_tensor : device_tensors) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    device_tensor->MoveTo(to, false, kDefaultStreamIndex);
  }
}

void MemorySwapActor::Run(OpContext<mindspore::runtime::DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  static std::map<device::SwapActionType, device::StorageType> swap_to_map = {
    {device::SwapActionType::kHBM2DDR, device::StorageType::kHost},
    {device::SwapActionType::kDDR2HBM, device::StorageType::kDevice},
    {device::SwapActionType::kDDR2DISK, device::StorageType::kFile},
    {device::SwapActionType::kDISK2DDR, device::StorageType::kHost},
    {device::SwapActionType::kHBM2DISK, device::StorageType::kFile},
    {device::SwapActionType::kDISK2HBM, device::StorageType::kDevice}};
  UpdateDeviceTensors(context);
  for (const auto &action : swap_actions_) {
    const auto action_type = action.first;
    const auto &device_tensor_indexes = action.second;
    const auto &device_tensors = GetDeviceTensors(device_tensor_indexes);
    if (action_type == device::SwapActionType::kAllocHBM) {
      AllocDeviceContinuousMem(device_tensors);
    } else if (action_type != device::SwapActionType::kUnDefined) {
      Swap(swap_to_map[action_type], device_tensors);
    } else {
      MS_LOG(WARNING) << "Unknown swap action type, skip.";
    }
  }
  EraseInput(context);
  SendOutput(context);
}

void MemorySwapInActor::Run(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  FetchRealParameters(context);
  if (continuous_device_tensors_.size() != continuous_device_tensor_sizes_.size()) {
    MS_LOG(EXCEPTION) << "Size of continuous_device_tensors_ and continuous_device_tensor_sizes_ should be same,"
                         " bug got"
                      << continuous_device_tensors_.size() << ", " << continuous_device_tensor_sizes_.size();
  }
  for (size_t j = 0; j < continuous_device_tensors_.size(); ++j) {
    const auto &device_ptrs =
      device_contexts_[0]->device_res_manager_->AllocateContinuousMemory(continuous_device_tensor_sizes_[j]);
    for (size_t k = 0; k < continuous_device_tensors_[j].size(); ++k) {
      continuous_device_tensors_[j][k]->set_ptr(device_ptrs[k]);
      continuous_device_tensors_[j][k]->set_from_mem_pool(true);
    }
  }
  for (const auto &device_tensor : device_tensors_to_swap_) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->mem_offloaded() && !device_tensor->Load(stream_id_)) {
      MS_LOG(EXCEPTION) << "Load device tensor from host to device failed.";
    }
  }
  for (const auto &real_parameter : real_parameters_) {
    MS_EXCEPTION_IF_NULL(real_parameter);
    if (real_parameter->mem_offloaded() && !real_parameter->Load(stream_id_)) {
      MS_LOG(EXCEPTION) << "Load device tensor from host to device failed.";
    }
  }
  EraseInput(context);
  SendOutput(context);
}

void MemorySwapOutActor::Run(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  FetchRealParameters(context);
  for (const auto &device_tensor : device_tensors_to_swap_) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->mem_offloaded() || device_tensor->GetPtr() == nullptr) {
      continue;
    }
    if (!device_tensor->Offload(stream_id_)) {
      MS_LOG(EXCEPTION) << "Offload device tensor from device to host failed.";
    }
  }
  for (const auto &device_tensor : device_tensors_to_free_) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    // Offload DeviceTensor with max original_ref_count which will not be used anymore in current sub graph.
    // DeviceTensor without max original_ref_count will be free in MemoryManagerActor::FreeMemoryByRefCount.
    if (device_tensor->mem_offloaded() || device_tensor->GetPtr() == nullptr ||
        device_tensor->original_ref_count() != SIZE_MAX) {
      continue;
    }
    if (!device_tensor->Offload(stream_id_)) {
      MS_LOG(EXCEPTION) << "Offload device tensor from device to host failed.";
    }
  }

  for (size_t i = 0; i < real_parameters_.size(); ++i) {
    const auto &real_parameter = real_parameters_[i];
    MS_EXCEPTION_IF_NULL(real_parameter);
    if (real_parameter->mem_offloaded() || real_parameter->GetPtr() == nullptr) {
      continue;
    }
    if (swap_out_real_parameter_[i] || real_parameter->original_ref_count() == SIZE_MAX) {
      if (!real_parameter->Offload(stream_id_)) {
        MS_LOG(EXCEPTION) << "Offload device tensor from device to host failed.";
      }
    }
  }
  EraseInput(context);
  SendOutput(context);
}
}  // namespace runtime
}  // namespace mindspore
