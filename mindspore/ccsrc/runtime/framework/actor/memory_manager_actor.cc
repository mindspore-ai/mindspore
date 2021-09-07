/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void MemoryManagerActor::AllocateMemory(const std::vector<DeviceTensor *> *alloc_list,
                                        const DeviceContext *device_context, OpContext<DeviceTensor> *const op_context,
                                        const AID from_aid) {
  MS_EXCEPTION_IF_NULL(alloc_list);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);

  for (auto &device_tensor : *alloc_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->GetPtr() != nullptr) {
      continue;
    }
    // Allocate memory through the device context.
    if (!device_context->AllocateMemory(device_tensor, device_tensor->GetSize())) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*op_context), device_context,
                                                  from_aid.Name(), device_tensor->GetSize());
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  Async(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}

void MemoryManagerActor::AllocateContinuousMemory(const std::vector<std::vector<DeviceTensorPtr>> *alloc_list_list,
                                                  const std::vector<std::vector<size_t>> *size_list_list,
                                                  const std::vector<size_t> *total_size_list,
                                                  const std::vector<const DeviceContext *> *device_contexts,
                                                  OpContext<DeviceTensor> *const op_context, const AID from_aid) {
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
    auto &total_size = (*total_size_list)[i];
    auto &device_context = (*device_contexts)[i];
    // Allocate memory through the device context.
    if (!device_context->AllocateContinuousMemory(alloc_list, total_size, size_list)) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*op_context), device_context,
                                                  from_aid.Name(), total_size);
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  Async(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}

void MemoryManagerActor::AllocateBatchMemory(const std::vector<DeviceTensor *> *alloc_list,
                                             const std::vector<const DeviceContext *> *device_contexts,
                                             OpContext<DeviceTensor> *const op_context, const AID from_aid) {
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
    if (device_tensor->GetPtr() != nullptr) {
      continue;
    }

    // Allocate memory through the device context.
    if (!device_context->AllocateMemory(device_tensor, device_tensor->GetSize())) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*op_context), device_context,
                                                  from_aid.Name(), device_tensor->GetSize());
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  Async(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}

void MemoryManagerActor::FreeMemory(const std::vector<DeviceTensor *> *free_list, const DeviceContext *device_context,
                                    OpContext<DeviceTensor> *) {
  MS_EXCEPTION_IF_NULL(free_list);
  MS_EXCEPTION_IF_NULL(device_context);
  for (auto &device_tensor : *free_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->original_ref_count() == SIZE_MAX) {
      continue;
    }
    // The reference count is decremented to zero to free memory, and reset to the original count.
    device_tensor->DecreaseRefCount();
    if (device_tensor->ref_count() == 0) {
      // Free memory through the device context.
      if (device_tensor->GetPtr() != nullptr) {
        device_context->FreeMemory(device_tensor);
      }
      device_tensor->ResetRefCount();
    }
  }
}

void MemoryManagerActor::FreeBatchMemory(const std::vector<DeviceTensor *> *free_list,
                                         const std::vector<const DeviceContext *> *device_contexts,
                                         OpContext<DeviceTensor> *const op_context) {
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
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_EXCEPTION_IF_NULL(device_context);
    if (device_tensor->original_ref_count() == SIZE_MAX) {
      continue;
    }
    // The reference count is decremented to zero to free memory, and reset to the original count.
    device_tensor->DecreaseRefCount();
    if (device_tensor->ref_count() == 0) {
      // Free memory through the device context.
      if (device_tensor->GetPtr() != nullptr) {
        device_context->FreeMemory(device_tensor);
      }
      device_tensor->ResetRefCount();
    }
  }
}

void MemoryManagerActor::Wait(OpContext<DeviceTensor> *const op_context, const AID from_aid) {
  // Call back to the from actor to process.
  Async(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}
}  // namespace runtime
}  // namespace mindspore
