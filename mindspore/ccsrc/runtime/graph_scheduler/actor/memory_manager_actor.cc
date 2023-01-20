/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/data_source_actor.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
namespace {
void OnMemoryAllocFinish(const AID &from_aid, OpContext<DeviceTensor> *const op_context) {
  if (!ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::Send(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
  }
}
}  // namespace

void MemoryManagerActor::AllocateMemory(const std::vector<DeviceTensor *> *alloc_list,
                                        const DeviceContext *device_context, OpContext<DeviceTensor> *const op_context,
                                        const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(alloc_list);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);

  for (auto &device_tensor : *alloc_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    // Unused device address need skip to reduce memory use.
    if (device_tensor->IsPtrValid() || TEST_FLAG(device_tensor->flag(), device::kDeviceAddressFlagNotUsed)) {
      continue;
    }

    try {
      // Allocate memory through the device context.
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
      if (!device_context->device_res_manager_->AllocateMemory(device_tensor)) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
      return;
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);
}

void MemoryManagerActor::AllocateContinuousMemory(const std::vector<std::vector<DeviceTensorPtr>> *alloc_list_list,
                                                  const std::vector<std::vector<size_t>> *size_list_list,
                                                  const std::vector<size_t> *total_size_list,
                                                  const std::vector<const DeviceContext *> *device_contexts,
                                                  OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(alloc_list_list);
  MS_EXCEPTION_IF_NULL(size_list_list);
  MS_EXCEPTION_IF_NULL(total_size_list);
  MS_EXCEPTION_IF_NULL(device_contexts);
  MS_EXCEPTION_IF_NULL(op_context);
  if (((*alloc_list_list).size() != (*size_list_list).size()) ||
      ((*size_list_list).size() != (*total_size_list).size()) ||
      ((*total_size_list).size() != (*device_contexts).size())) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR(
      (*op_context), "The size of alloc_list_list, size_list_list, total_size_list and device_contexts are not equal.");
  }

  for (size_t i = 0; i < (*alloc_list_list).size(); ++i) {
    auto &alloc_list = (*alloc_list_list)[i];
    auto &size_list = (*size_list_list)[i];
    auto &device_context = (*device_contexts)[i];
    MS_EXCEPTION_IF_NULL(device_context);
    // If the address of continuous tensor has already been allocated, skip the tensor.
    if (alloc_list[0]->GetPtr() != nullptr) {
      MS_LOG(WARNING) << "The continuous memory has already been allocated of actor: " << from_aid.Name()
                      << " with index: " << i;
      continue;
    }
    // Allocate memory through the device context.
    device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
    auto dev_ptr_list = device_context->device_res_manager_->AllocateContinuousMemory(size_list);
    if (dev_ptr_list.empty() || dev_ptr_list.size() != alloc_list.size()) {
      MS_LOG(ERROR) << "Allocate continuous memory failed, device ptr list size: " << dev_ptr_list.size()
                    << ", address list size:" << alloc_list.size();
      auto &total_size = (*total_size_list)[i];
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, total_size, op_context);
      return;
    }

    for (size_t index = 0; index < alloc_list.size(); index++) {
      MS_EXCEPTION_IF_NULL(alloc_list[index]);
      if (alloc_list[index]->GetPtr() != nullptr) {
        auto old_dev_addr = alloc_list[index];
        MS_EXCEPTION_IF_NULL(old_dev_addr);
        auto old_size = old_dev_addr->GetSize();
        if (old_size > size_list[index]) {
          MS_LOG(EXCEPTION) << "Device size of old device address is larger than new device address, " << old_size
                            << " vs " << size_list[index];
        }
        auto new_dev_addr = device_context->device_res_manager_->CreateDeviceAddress(
          dev_ptr_list[index], old_size, old_dev_addr->format(), old_dev_addr->type_id(), old_dev_addr->host_shape());
        (void)new_dev_addr->SyncDeviceToDevice(old_dev_addr.get());
        device_context->device_res_manager_->FreeMemory(old_dev_addr.get());
      }
      alloc_list[index]->set_ptr(dev_ptr_list[index]);
      alloc_list[index]->SetSize(size_list[index]);
      alloc_list[index]->set_from_mem_pool(true);
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);
}

void MemoryManagerActor::AllocateBatchMemory(const std::vector<DeviceTensor *> *alloc_list,
                                             const std::vector<const DeviceContext *> *device_contexts,
                                             OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(alloc_list);
  MS_EXCEPTION_IF_NULL(device_contexts);
  MS_EXCEPTION_IF_NULL(op_context);
  if ((*alloc_list).size() != (*device_contexts).size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context),
                                      "The size of alloc list is not equal to the size of device contexts.");
  }

  for (size_t i = 0; i < (*alloc_list).size(); ++i) {
    auto &device_tensor = (*alloc_list)[i];
    auto &device_context = (*device_contexts)[i];
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_EXCEPTION_IF_NULL(device_context);
    // Unused device address need skip to reduce memory use.
    if (device_tensor->IsPtrValid() || TEST_FLAG(device_tensor->flag(), device::kDeviceAddressFlagNotUsed)) {
      continue;
    }

    try {
      // Allocate memory through the device context.
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
      if (!device_context->device_res_manager_->AllocateMemory(device_tensor)) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
      return;
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);
}

void MemoryManagerActor::AllocateSomasMemory(SomasInfo *const somas_info, const DeviceContext *device_context,
                                             OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(somas_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(op_context);

  // Allocate the whole block memory.
  if (somas_info->base_address_ != nullptr) {
    std::string error_info = from_aid.Name() + " already has the base somas address.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
  }
  try {
    device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
    auto device_ptr = device_context->device_res_manager_->AllocateMemory(somas_info->whole_block_size_);
    if (device_ptr == nullptr) {
      MS_LOG(WARNING) << from_aid.Name()
                      << " allocate somas whole block memory failed, alloc size: " << somas_info->whole_block_size_
                      << ". Try to allocate the merged blocks memory.";
    } else {
      somas_info->base_address_ = device_ptr;
      return;
    }
  } catch (const std::exception &e) {
    SetOpContextMemoryAllocFail(from_aid.Name(), device_context, somas_info->whole_block_size_, op_context);
    return;
  }

  // Allocate the merged blocks memory.
  try {
    auto &merged_base_addresses = somas_info->merged_base_addresses_;
    for (auto &megred_block : somas_info->merged_blocks_map_) {
      size_t block_offset = megred_block.first;
      size_t block_size = megred_block.second;
      if ((merged_base_addresses.count(block_offset) > 0) && (merged_base_addresses[block_offset] != nullptr)) {
        std::string error_info = from_aid.Name() + " already has the base somas address.";
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
      }
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
      auto device_ptr = device_context->device_res_manager_->AllocateMemory(block_size);
      if (device_ptr == nullptr) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, block_size, op_context);
        return;
      }
      merged_base_addresses[block_offset] = device_ptr;
    }
  } catch (const std::exception &e) {
    SetOpContextMemoryAllocFail(from_aid.Name(), device_context, somas_info->whole_block_size_, op_context);
    return;
  }
  MS_LOG(WARNING) << from_aid.Name() << " allocate somas merged blocks memory succeeded and continue running.";

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);
}

void MemoryManagerActor::FreeMemory(const std::vector<DeviceTensor *> *free_list, const DeviceContext *device_context,
                                    OpContext<DeviceTensor> *, const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(free_list);
  for (auto &device_tensor : *free_list) {
    FreeMemoryByRefCount(device_tensor, device_context, from_aid.Name());
  }
}

void MemoryManagerActor::FreeBatchMemory(const std::vector<DeviceTensor *> *free_list,
                                         const std::vector<const DeviceContext *> *device_contexts,
                                         OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(free_list);
  MS_EXCEPTION_IF_NULL(device_contexts);
  MS_EXCEPTION_IF_NULL(op_context);
  if ((*free_list).size() != (*device_contexts).size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context),
                                      "The size of free list is not equal to the size of device contexts.");
  }

  for (size_t i = 0; i < (*free_list).size(); ++i) {
    auto &device_tensor = (*free_list)[i];
    auto &device_context = (*device_contexts)[i];
    FreeMemoryByRefCount(device_tensor, device_context, from_aid.Name());
  }
}

void MemoryManagerActor::FreeSomasMemory(SomasInfo *const somas_info, const DeviceContext *device_context,
                                         OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(somas_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(op_context);

  // Free the whole block memory.
  if (somas_info->base_address_ != nullptr) {
    device_context->device_res_manager_->FreeMemory(somas_info->base_address_);
    somas_info->base_address_ = nullptr;

    for (auto &merged_base_address : somas_info->merged_base_addresses_) {
      if (merged_base_address.second != nullptr) {
        std::string error_info = " There should have no megred block base address for " + from_aid.Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
      }
    }
    return;
  }

  // Free the merged blocks memory.
  for (auto &merged_base_address : somas_info->merged_base_addresses_) {
    if (merged_base_address.second == nullptr) {
      std::string error_info = " There should have megred block base address for " + from_aid.Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
    }
    device_context->device_res_manager_->FreeMemory(merged_base_address.second);
    merged_base_address.second = nullptr;
  }
}

void MemoryManagerActor::Wait(OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  // Call back to the from actor to process.
  ActorDispatcher::Send(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}

// Only one of the static and dynamic reference counts will take effect.
void MemoryManagerActor::FreeMemoryByRefCount(DeviceTensor *const device_tensor, const DeviceContext *device_context,
                                              const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(device_tensor);

  std::lock_guard<std::mutex> locker(mem_free_mutex_);
  if (device_tensor->original_ref_count() != SIZE_MAX) {
    // The static reference count is decremented to zero to free memory, and reset to the original count.
    device_tensor->DecreaseRefCount();
    if (device_tensor->ref_count() == 0) {
      device_tensor->ResetRefCount();
      device_tensor->ClearUserData();
      if (device_tensor->GetPtr() != nullptr) {
        auto held_by_nodes = device_tensor->held_by_nodes();
        if (held_by_nodes.empty()) {
          FreeMemoryByDeviceContext(device_tensor, device_context);
        } else {
          FreeMemoryByValueNode(held_by_nodes, device_tensor);
        }
      }
    }
  } else if (device_tensor->dynamic_ref_count() != INT32_MAX) {
    // The dynamic reference count is decremented to zero to free memory.
    device_tensor->DecreaseDynamicRefCount(op_name);
    if ((device_tensor->dynamic_ref_count() == 0) && (device_tensor->GetPtr() != nullptr)) {
      device_tensor->ClearUserData();
      MS_LOG(DEBUG) << "Free memory by the dynamic reference count, device address" << device_tensor->GetPtr();
      FreeMemoryByDeviceContext(device_tensor, device_context);
    }
  }
}

void MemoryManagerActor::SetOpContextMemoryAllocFail(const std::string &kernel_name,
                                                     const DeviceContext *device_context, size_t alloc_size,
                                                     OpContext<DeviceTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);

  std::lock_guard<std::mutex> locker(mem_alloc_failed_mutex_);
  int step_id = op_context->sequential_num_;
  // First occur allocating memory failed.
  if (mem_alloc_failed_step_ids_.find(step_id) == mem_alloc_failed_step_ids_.end()) {
    mem_alloc_failed_step_ids_.clear();
    (void)mem_alloc_failed_step_ids_.insert(step_id);
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *op_context, *device_context,
                                                kernel_name, alloc_size);
  }
}
}  // namespace runtime
}  // namespace mindspore
