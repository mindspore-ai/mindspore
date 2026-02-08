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

#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "backend/ms_backend/runtime/actors/base/data_source_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_actor.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace runtime {
namespace {
void OnMemoryAllocFinish(const AID &from_aid, OpContext<KernelTensor> *const op_context) {
  if (!ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::Send(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
  }
}
}  // namespace

void MemoryManagerActor::AllocateMemory(const std::vector<KernelTensorPtr> *alloc_list,
                                        const DeviceContext *device_context, OpContext<KernelTensor> *const op_context,
                                        const AID &from_aid) {
  for (auto &kernel_tensor : *alloc_list) {
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_tensor = kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(device_tensor);
    // Unused device address need skip to reduce memory use.
    if (kernel_tensor->IsNotNeedAlloc()) {
      continue;
    }

    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(), memory::mem_pool::MemType::kKernel,
                                                   device_tensor->GetSize(), device_tensor);

    try {
      bool success = false;
      if (kernel_tensor->continuous_kernel_tensors() == nullptr) {
        success = device_context->device_res_manager_->AllocateMemory(device_tensor, kernel_tensor->alloc_stream_id());
        if (success) {
          static std::string name = "Alloc memory";
          kernel_tensor->IncreaseNewRefCount(name);
        }
      } else {
        device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "ContinuousMemory", "", false);

        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Allocate continuous memory, device address : " << device_tensor << ".";
        success = AllocateContinuousMemory(kernel_tensor.get(), device_context, from_aid);
      }

      if (!success) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
      return;
    }

    if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
      auto output_address = reinterpret_cast<std::uintptr_t>(device_tensor);
      MS_LOG(WARNING) << "Need Profile Memory, alloc type: MemoryManagerActor, device address class ptr: "
                      << output_address << ", device address size: " << device_tensor->GetSize()
                      << ", device address addr: " << device_tensor->GetPtr();
    }
  }
}

void MemoryManagerActor::AllocateMemoryHP(const std::vector<KernelTensorPtr> *alloc_list,
                                          const DeviceContext *device_context,
                                          OpContext<KernelTensor> *const op_context, const AID &from_aid,
                                          uint32_t mem_alloc_stream_id) {
  for (auto &kernel_tensor : *alloc_list) {
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_tensor = kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(device_tensor);
    // Unused device address need skip to reduce memory use.
    if (kernel_tensor->IsNotNeedAllocWOLock()) {
      continue;
    }
    try {
      bool success = false;
      if (kernel_tensor->continuous_kernel_tensors() == nullptr) {
        // When stream_id is kInValidStreamIndex, the alloc stream ID of kernel_tensor is used.
        auto alloc_stream_id =
          (mem_alloc_stream_id == kInValidStreamIndex) ? kernel_tensor->alloc_stream_id() : mem_alloc_stream_id;
        success = device_context->device_res_manager_->AllocateMemory(device_tensor, alloc_stream_id);
        if (success) {
          static std::string name = "Alloc memory";
          kernel_tensor->IncreaseNewRefCount(name);
        }
      } else {
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Allocate continuous memory, device address : " << device_tensor << ".";
        success = AllocateContinuousMemory(kernel_tensor.get(), device_context, from_aid);
      }

      if (!success) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
      return;
    }
  }
}

bool MemoryManagerActor::AllocateContinuousMemory(KernelTensor *kernel_tensor, const DeviceContext *device_context,
                                                  const AID &from_aid) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(device_tensor);
  std::vector<size_t> size_list;
  const auto &continuous_kernel_tensors = kernel_tensor->continuous_kernel_tensors();
  for (const auto &continuous_kernel_tensor_wpr : *continuous_kernel_tensors) {
    const auto &continuous_kernel_tensor = continuous_kernel_tensor_wpr.lock();
    MS_EXCEPTION_IF_NULL(continuous_kernel_tensor);
    auto device_address = continuous_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    (void)size_list.emplace_back(device_address->GetSize());
  }
  const auto &device_addresses =
    device_context->device_res_manager_->AllocateContinuousMemory(size_list, kDefaultStreamIndex);
  if (device_addresses.size() == continuous_kernel_tensors->size()) {
    for (size_t i = 0, end = (*continuous_kernel_tensors).size(); i < end; ++i) {
      const auto &continuous_kernel_tensor = (*(continuous_kernel_tensors))[i].lock();
      MS_EXCEPTION_IF_NULL(continuous_kernel_tensor);
      auto device_address = continuous_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_address);
      MS_EXCEPTION_IF_CHECK_FAIL(device_address->GetPtr() == nullptr, "Continuous memory conflicted.");
      device_address->set_ptr(device_addresses[i]);
      device_address->set_from_mem_pool(true);
      continuous_kernel_tensor->IncreaseNewRefCount(from_aid.Name() + " alloc continue memory");
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(),
                                                     memory::mem_pool::MemType::kContinuousMemory,
                                                     device_address->GetSize(), device_tensor);
      if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
        device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, device_tensor, device_addresses[i]);
      }
    }
    return true;
  }
  return false;
}

void MemoryManagerActor::AllocateContinuousMemory(const std::vector<std::vector<KernelTensorPtr>> *alloc_list_list,
                                                  const std::vector<std::vector<size_t>> *size_list_list,
                                                  const std::vector<uint32_t> *stream_id_list,
                                                  const std::vector<size_t> *total_size_list,
                                                  const std::vector<const DeviceContext *> *device_contexts,
                                                  OpContext<KernelTensor> *const op_context, const AID &from_aid) {
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

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "ContinuousMemory", "", false);
  for (size_t i = 0; i < (*alloc_list_list).size(); ++i) {
    auto &alloc_list = (*alloc_list_list)[i];
    auto &size_list = (*size_list_list)[i];
    auto stream_id = (*stream_id_list)[i];
    auto &device_context = (*device_contexts)[i];
    MS_EXCEPTION_IF_NULL(device_context);
    // If the address of continuous tensor has already been allocated, skip the tensor.
    if (alloc_list[0]->device_ptr() != nullptr) {
      MS_LOG(WARNING) << "The continuous memory has already been allocated of actor: " << from_aid.Name()
                      << " with index: " << i;
      continue;
    }
    // Allocate memory through the device context.
    device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), memory::mem_pool::MemType::kKernel);
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
      auto &old_dev_kernel_tensor = alloc_list[index];
      MS_EXCEPTION_IF_NULL(old_dev_kernel_tensor);
      auto &old_dev_addr = old_dev_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(old_dev_addr);
      if (old_dev_addr->GetPtr() != nullptr) {
        auto old_size = old_dev_addr->GetSize();
        if (old_size > size_list[index]) {
          MS_LOG(EXCEPTION) << "Device size of old device address is larger than new device address, " << old_size
                            << " vs " << size_list[index];
        }

        auto kernel_tensor =
          AnfAlgo::CreateKernelTensor(dev_ptr_list[index], old_size, old_dev_kernel_tensor->format(),
                                      old_dev_kernel_tensor->dtype_id(), old_dev_kernel_tensor->GetShapeVector(),
                                      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                      device_context->device_context_key().device_id_);
        kernel_tensor->set_stream_id(old_dev_addr->stream_id());
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Create kernel tensor:" << kernel_tensor->ToString();
        (void)SyncCopy(kernel_tensor.get(), old_dev_kernel_tensor.get(), stream_id);
        if (!device_context->device_res_manager_->SyncAllStreams()) {
          MS_LOG(ERROR) << "Failed to sync stream.";
        }
        device_context->device_res_manager_->FreeMemory(old_dev_addr.get());
      }
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(),
                                                     memory::mem_pool::MemType::kContinuousMemory,
                                                     old_dev_addr->GetSize(), old_dev_addr.get());
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, old_dev_addr.get(), dev_ptr_list[index]);
      old_dev_addr->set_ptr(dev_ptr_list[index]);
      old_dev_addr->SetSize(size_list[index]);
      old_dev_addr->set_from_mem_pool(true);
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, from_aid.Name(), false);
}

void MemoryManagerActor::AllocateBatchMemory(const std::vector<KernelTensorPtr> *alloc_list,
                                             const std::vector<const DeviceContext *> *device_contexts,
                                             OpContext<KernelTensor> *const op_context, const AID &from_aid) {
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
    auto &kernel_tensor = (*alloc_list)[i];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_tensor = kernel_tensor->device_address().get();
    auto &device_context = (*device_contexts)[i];
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_EXCEPTION_IF_NULL(device_context);
    // Unused device address need skip to reduce memory use.
    if (kernel_tensor->IsNotNeedAlloc()) {
      continue;
    }

    try {
      // Allocate memory through the device context.
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "BatchMemory", "", false);
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
        AddMemInfo, from_aid.Name(), memory::mem_pool::MemType::kBatchMemory, device_tensor->GetSize(), device_tensor);
      if (!device_context->device_res_manager_->AllocateMemory(device_tensor, kDefaultStreamIndex)) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
      static std::string name = "Alloc memory";
      kernel_tensor->IncreaseNewRefCount(name);
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
                                             OpContext<KernelTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  MS_EXCEPTION_IF_NULL(somas_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(op_context);

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "SomasMemory",
                                                 "kernel_graph_" + std::to_string(somas_info->graph_id_), false);

  // Allocate the whole block memory.
  if (somas_info->base_address_ != nullptr) {
    std::string error_info = from_aid.Name() + " already has the base somas address.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
  }

  if (memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocSomasWholeBlock)) {
    try {
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), memory::mem_pool::MemType::kKernel);
      auto device_ptr = device_context->device_res_manager_->AllocateMemory(somas_info->whole_block_size_);
      if (device_ptr == nullptr) {
        MS_LOG(INFO) << from_aid.Name()
                     << " allocate somas whole block memory failed, alloc size: " << somas_info->whole_block_size_
                     << ". Try to allocate the merged blocks memory.";
      } else {
        device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, from_aid.Name(),
                                                       somas_info->whole_block_size_, device_ptr,
                                                       memory::mem_pool::MemType::kSomas);
        somas_info->base_address_ = device_ptr;
        PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, from_aid.Name(), false);
        MS_LOG(INFO) << from_aid.Name() << " allocate somas whole block memory succeeded and continue running.";
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, somas_info->whole_block_size_, op_context);
      return;
    }
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
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(from_aid.Name(), memory::mem_pool::MemType::kKernel);
      auto device_ptr = device_context->device_res_manager_->AllocateMemory(block_size);
      if (device_ptr == nullptr) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, block_size, op_context);
        return;
      }
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, from_aid.Name(), block_size, device_ptr,
                                                     memory::mem_pool::MemType::kSomas);
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

void MemoryManagerActor::FreeMemory(const std::vector<KernelTensorPtr> *free_list, const DeviceContext *device_context,
                                    OpContext<KernelTensor> *, const AID &from_aid) {
  for (auto &kernel_tensor : *free_list) {
    FreeMemoryByRefCount(kernel_tensor.get(), device_context, from_aid.Name());
  }
}

void MemoryManagerActor::FreeBatchMemory(const std::vector<KernelTensorPtr> *free_list,
                                         const std::vector<const DeviceContext *> *device_contexts,
                                         OpContext<KernelTensor> *const op_context, const AID &from_aid) {
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
    auto &kernel_tensor = (*free_list)[i];
    auto &device_context = (*device_contexts)[i];
    FreeMemoryByRefCount(kernel_tensor.get(), device_context, from_aid.Name());
  }

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryFree, from_aid.Name(), false);
}

void MemoryManagerActor::FreeSomasMemory(SomasInfo *const somas_info, const DeviceContext *device_context,
                                         OpContext<KernelTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  MS_EXCEPTION_IF_NULL(somas_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  MS_EXCEPTION_IF_NULL(op_context);

  std::vector<void *> keep_addrs;
  for (auto &output_address : somas_info->graph_output_device_addresses_) {
    MS_EXCEPTION_IF_NULL(output_address);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Keep address:" << output_address << " ptr:" << output_address->GetPtr()
      << " size:" << output_address->GetSize() << " for actor:" << from_aid;
    (void)keep_addrs.emplace_back(output_address->GetMutablePtr());
  }

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
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "SomasOutput", "", false);
  for (auto &output_address : somas_info->graph_output_device_addresses_) {
    output_address->set_from_mem_pool(true);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Set from mem pool for device address:" << output_address;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(), memory::mem_pool::MemType::kSomasOutput,
                                                   output_address->GetSize(), output_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, output_address, output_address->GetPtr());
  }

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryFree, from_aid.Name(), false);
}

void MemoryManagerActor::Wait(OpContext<KernelTensor> *const op_context, const AID &from_aid) {
  // Call back to the from actor to process.
  ActorDispatcher::Send(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}

// Only one of the static and dynamic reference counts will take effect.
void MemoryManagerActor::FreeMemoryByRefCount(KernelTensor *const kernel_tensor, const DeviceContext *device_context,
                                              const std::string &op_name) {
  if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
    return;
  }
  const auto &device_tensor = kernel_tensor->device_address().get();
  if (kernel_tensor->new_ref_count() != SIZE_MAX) {
    if (kernel_tensor->new_ref_count() == 0) {
      const auto &node_with_index = device_tensor->GetNodeIndex();
      MS_LOG(EXCEPTION) << "Invalid new ref count:0 for decrease for kernel tensor:" << kernel_tensor->ToString()
                        << " node:"
                        << (node_with_index.first == nullptr
                              ? "null"
                              : node_with_index.first->fullname_with_scope() +
                                  " debug string:" + node_with_index.first->DebugString())
                        << " index:" << node_with_index.second << " actor:" << op_name;
    }

    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Op:" << op_name << " decrease new ref count for:" << kernel_tensor->ToString();
    if ((kernel_tensor->DecreaseNewRefCount(op_name) == 0) && device_tensor->IsPtrValid()) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Op:" << op_name << " free memory by the new reference count, kernel tensor:" << kernel_tensor->ToString()
        << ".";
      auto held_by_nodes = device_tensor->held_by_nodes();
      if (held_by_nodes.empty()) {
        FreeMemoryByDeviceContext(device_tensor, device_context);
      } else {
        FreeMemoryByValueNode(held_by_nodes, device_tensor);
      }
    }
  }
}

void MemoryManagerActor::SetOpContextMemoryAllocFail(const std::string &kernel_name,
                                                     const DeviceContext *device_context, size_t alloc_size,
                                                     OpContext<KernelTensor> *const op_context) {
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
