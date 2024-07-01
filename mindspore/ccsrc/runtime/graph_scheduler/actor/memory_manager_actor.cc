/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#include "include/backend/mem_reuse/mem_tracker.h"
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
  for (auto &device_tensor : *alloc_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    // Unused device address need skip to reduce memory use.
    if (device_tensor->IsNotNeedAlloc()) {
      continue;
    }

    if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(), device::tracker::MemType::kKernel,
                                                     device_tensor->GetSize(), device_tensor);
    }

    try {
      // Allocate memory through the device context.
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
      if (!device_context->device_res_manager_->AllocateMemory(device_tensor, kDefaultStreamIndex)) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
      return;
    }

    if (common::IsNeedProfileMemory()) {
      auto output_address = reinterpret_cast<std::uintptr_t>(device_tensor);
      MS_LOG(WARNING) << "Need Profile Memory, alloc type: MemoryManagerActor, device address class ptr: "
                      << output_address << ", device address size: " << device_tensor->GetSize()
                      << ", device address addr: " << device_tensor->GetPtr();
    }
  }
}

void MemoryManagerActor::AllocateContinuousMemory(const std::vector<std::vector<DeviceTensorPtr>> *alloc_list_list,
                                                  const std::vector<std::vector<size_t>> *size_list_list,
                                                  const std::vector<uint32_t> *stream_id_list,
                                                  const std::vector<size_t> *total_size_list,
                                                  const std::vector<const DeviceContext *> *device_contexts,
                                                  OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  MS_EXCEPTION_IF_NULL(alloc_list_list);
  MS_EXCEPTION_IF_NULL(size_list_list);
  MS_EXCEPTION_IF_NULL(total_size_list);
  MS_EXCEPTION_IF_NULL(device_contexts);
  MS_EXCEPTION_IF_NULL(op_context);
  if (((*alloc_list_list).size() != (*size_list_list).size()) ||
      ((*size_list_list).size() != (*stream_id_list).size()) ||
      ((*stream_id_list).size() != (*total_size_list).size()) ||
      ((*total_size_list).size() != (*device_contexts).size())) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context),
                                      "The size of alloc_list_list, size_list_list, stream_id_list, total_size_list "
                                      "and device_contexts are not equal.");
  }

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "ContinuousMemory", "");
  for (size_t i = 0; i < (*alloc_list_list).size(); ++i) {
    auto &alloc_list = (*alloc_list_list)[i];
    auto &size_list = (*size_list_list)[i];
    auto stream_id = (*stream_id_list)[i];
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
    auto dev_ptr_list = device_context->device_res_manager_->AllocateContinuousMemory(size_list, stream_id);
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

        auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
          dev_ptr_list[index], old_size, kernel::GetFormatFromStrToEnum(old_dev_addr->format()),
          old_dev_addr->type_id(), old_dev_addr->host_shape(), device_context->device_context_key().device_name_,
          device_context->device_context_key().device_id_);
        kernel_tensor->set_stream_id(old_dev_addr->stream_id());
        auto new_dev_addr = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
        MS_LOG(DEBUG) << "Create device tensor:" << new_dev_addr << " type:" << new_dev_addr->type_id();
        (void)new_dev_addr->SyncDeviceToDevice(old_dev_addr.get());
        device_context->device_res_manager_->FreeMemory(old_dev_addr.get());
      }
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(),
                                                     device::tracker::MemType::kContinuousMemory,
                                                     alloc_list[index]->GetSize(), alloc_list[index].get());
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, alloc_list[index].get(), dev_ptr_list[index]);
      alloc_list[index]->set_ptr(dev_ptr_list[index]);
      alloc_list[index]->SetSize(size_list[index]);
      alloc_list[index]->set_from_mem_pool(true);
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, from_aid.Name(), false);
}

void MemoryManagerActor::AllocateBatchMemory(const std::vector<DeviceTensor *> *alloc_list,
                                             const std::vector<const DeviceContext *> *device_contexts,
                                             OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

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
    if (device_tensor->IsNotNeedAlloc()) {
      continue;
    }

    try {
      // Allocate memory through the device context.
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "BatchMemory", "");
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
        AddMemInfo, from_aid.Name(), device::tracker::MemType::kBatchMemory, device_tensor->GetSize(), device_tensor);
      if (!device_context->device_res_manager_->AllocateMemory(device_tensor, kDefaultStreamIndex)) {
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

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, from_aid.Name(), false);
}

void MemoryManagerActor::AllocateSomasMemory(SomasInfo *const somas_info, const DeviceContext *device_context,
                                             OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  MS_EXCEPTION_IF_NULL(somas_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(op_context);

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "SomasMemory",
                                                 "kernel_graph_" + std::to_string(somas_info->graph_id_));

  // Allocate the whole block memory.
  if (somas_info->base_address_ != nullptr) {
    std::string error_info = from_aid.Name() + " already has the base somas address.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
  }
  try {
    device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kKernelOutput);
    auto device_ptr = device_context->device_res_manager_->AllocateMemory(somas_info->whole_block_size_);
    if (device_ptr == nullptr) {
      MS_LOG(INFO) << from_aid.Name()
                   << " allocate somas whole block memory failed, alloc size: " << somas_info->whole_block_size_
                   << ". Try to allocate the merged blocks memory.";
    } else {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, from_aid.Name(),
                                                     somas_info->whole_block_size_, device_ptr,
                                                     device::tracker::MemType::kSomas);
      somas_info->base_address_ = device_ptr;
      PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, from_aid.Name(), false);
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
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, from_aid.Name(), block_size, device_ptr,
                                                     device::tracker::MemType::kSomas);
      merged_base_addresses[block_offset] = device_ptr;
    }
  } catch (const std::exception &e) {
    SetOpContextMemoryAllocFail(from_aid.Name(), device_context, somas_info->whole_block_size_, op_context);
    return;
  }
  MS_LOG(INFO) << from_aid.Name() << " allocate somas merged blocks memory succeeded and continue running.";

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, from_aid.Name(), false);
}

void MemoryManagerActor::FreeMemory(const std::vector<DeviceTensor *> *free_list, const DeviceContext *device_context,
                                    OpContext<DeviceTensor> *, const AID &from_aid) {
  for (auto &device_tensor : *free_list) {
    FreeMemoryByRefCount(device_tensor, device_context, from_aid.Name());
  }
}

void MemoryManagerActor::FreeBatchMemory(const std::vector<DeviceTensor *> *free_list,
                                         const std::vector<const DeviceContext *> *device_contexts,
                                         OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

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

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryFree, from_aid.Name(), false);
}

void MemoryManagerActor::FreeSomasMemory(SomasInfo *const somas_info, const DeviceContext *device_context,
                                         OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  MS_EXCEPTION_IF_NULL(somas_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(op_context);

  std::vector<void *> keep_addrs;
  for (auto &output_address : somas_info->graph_output_device_addresses_) {
    MS_EXCEPTION_IF_NULL(output_address);
    MS_LOG(DEBUG) << "Keep address:" << output_address << " ptr:" << output_address->GetPtr()
                  << " size:" << output_address->GetSize() << " for actor:" << from_aid;
    (void)keep_addrs.emplace_back(output_address->GetMutablePtr());
  }

  device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), device::AllocatorType::kGraphOutput);
  // Free the whole block memory.
  if (somas_info->base_address_ != nullptr) {
    device_context->device_res_manager_->FreePartMemorys({somas_info->base_address_}, keep_addrs,
                                                         somas_info->graph_output_address_sizes_);
    somas_info->base_address_ = nullptr;

    for (auto &merged_base_address : somas_info->merged_base_addresses_) {
      if (merged_base_address.second != nullptr) {
        std::string error_info = " There should have no megred block base address for " + from_aid.Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
      }
    }
  } else {
    // Free the merged blocks memory.
    std::vector<void *> free_addrs;
    for (auto &merged_base_address : somas_info->merged_base_addresses_) {
      if (merged_base_address.second == nullptr) {
        std::string error_info = " There should have megred block base address for " + from_aid.Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
      }
      (void)free_addrs.emplace_back(merged_base_address.second);
      merged_base_address.second = nullptr;
    }
    device_context->device_res_manager_->FreePartMemorys(free_addrs, keep_addrs,
                                                         somas_info->graph_output_address_sizes_);
  }

  // Somas decrease the ref count.
  for (auto &output_address : somas_info->graph_output_device_addresses_) {
    output_address->set_from_mem_pool(true);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(UpdateMemInfo, output_address,
                                                   device::tracker::MemType::kSomasOutput);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, output_address, output_address->GetPtr());
    FreeMemoryByRefCount(output_address, device_context, from_aid.Name());
  }

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryFree, from_aid.Name(), false);
}

void MemoryManagerActor::Wait(OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  // Call back to the from actor to process.
  ActorDispatcher::Send(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}

// Only one of the static and dynamic reference counts will take effect.
void MemoryManagerActor::FreeMemoryByRefCount(DeviceTensor *const device_tensor, const DeviceContext *device_context,
                                              const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (device_tensor->original_ref_count() != SIZE_MAX) {
    // The static reference count is decremented to zero to free memory, and reset to the original count.
    size_t ref_count = device_tensor->DecreaseRefCount();
    if (ref_count == 0) {
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
    if ((device_tensor->DecreaseDynamicRefCount(op_name) == 0) && (device_tensor->GetPtr() != nullptr)) {
      device_tensor->ClearUserData();
      MS_LOG(DEBUG) << "Free memory by the dynamic reference count, device address" << device_tensor->GetPtr() << ".";
      if (device_tensor->deleter() != nullptr) {
        MS_LOG(DEBUG) << "Free ptr:" << device_tensor->GetPtr() << " for device address:" << device_tensor;
        device_tensor->deleter()(static_cast<uint8_t *>(device_tensor->GetMutablePtr()));
        device_tensor->set_deleter(nullptr);
        device_tensor->set_ptr(nullptr);
        return;
      }
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
