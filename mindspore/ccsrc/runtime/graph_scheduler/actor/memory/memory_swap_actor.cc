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
      continuous_device_tensors_[j][k]->set_from_mem_pool(device_ptrs[k]);
    }
  }
  for (const auto &device_tensor : device_tensors_to_swap_) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->mem_offloaded()) {
      device_tensor->Load(stream_id_);
    }
  }
  for (const auto &real_parameter : real_parameters_) {
    MS_EXCEPTION_IF_NULL(real_parameter);
    if (real_parameter->mem_offloaded()) {
      real_parameter->Load(stream_id_);
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
    device_tensor->Offload(stream_id_);
  }
  for (const auto &device_tensor : device_tensors_to_free_) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->mem_offloaded() || device_tensor->GetPtr() == nullptr ||
        // Offload DeviceTensor with max original_ref_count which will not be used anymore in current sub graph.
        // DeviceTensor without max original_ref_count will be free in MemoryManagerActor::FreeMemoryByRefCount.
        device_tensor->original_ref_count() != SIZE_MAX) {
      continue;
    }
    device_tensor->Offload(stream_id_);
  }

  for (size_t i = 0; i < real_parameters_.size(); ++i) {
    const auto &real_parameter = real_parameters_[i];
    MS_EXCEPTION_IF_NULL(real_parameter);
    if (real_parameter->mem_offloaded() || real_parameter->GetPtr() == nullptr) {
      continue;
    }
    if (swap_out_real_parameter_[i] || real_parameter->original_ref_count() == SIZE_MAX) {
      real_parameter->Offload(stream_id_);
    }
  }
  EraseInput(context);
  SendOutput(context);
}
}  // namespace runtime
}  // namespace mindspore
