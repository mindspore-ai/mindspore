/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/utils/sliding_window/device_space_manager.h"

#include "device_address/device_address_maker.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/pipeline/pipeline.h"
#include "ir/tensor_new.h"
#include "mindspore/ccsrc/utils/sliding_window/sliding_window_internal.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace utils {
namespace sliding_window {

namespace {
device::DeviceContext *GetAscendDeviceContext() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey key = {device::DeviceType::kAscend, device_id};
  return device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(key);
}

TensorPtr SwapTensorToDevice(const TensorPtr &host_tensor, uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  auto device_addr =
    MakeDeviceAddress(host_tensor->data_type(), host_tensor->shape(), false, device::DeviceType::kAscend);
  MS_EXCEPTION_IF_NULL(device_addr);
  device_addr->set_stream_id(stream_id);
  if (!ctx->device_res_manager_->AllocateMemory(device_addr.get(), stream_id)) {
    MS_LOG(ERROR) << "Allocate device memory failed for tensor: " << host_tensor->ToString();
    return nullptr;
  }
  auto device_tensor = std::make_shared<tensor::Tensor>(host_tensor->data_type(), host_tensor->shape(), device_addr);
  auto host_addr = host_tensor->device_address();
  MS_EXCEPTION_IF_NULL(host_addr);
  if (!ctx->device_res_manager_->SyncCopy(device_addr, host_addr, stream_id, nullptr, nullptr)) {
    MS_LOG(ERROR) << "SyncCopy H2D failed for tensor: " << host_tensor->ToString();
    device_addr->ClearDeviceMemory();
    return nullptr;
  }
  return device_tensor;
}

TensorPtr SwapTensorToHost(const TensorPtr &device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  auto device_addr = device_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  auto host_tensor = tensor::from_spec(device_tensor->data_type(), device_tensor->shape(), device::DeviceType::kCPU);
  auto host_addr = host_tensor->device_address();
  MS_EXCEPTION_IF_NULL(host_addr);
  auto stream_id = device_addr->stream_id();
  if (!ctx->device_res_manager_->SyncCopy(host_addr, device_addr, stream_id, nullptr, nullptr)) {
    MS_LOG(ERROR) << "SyncCopy D2H failed for tensor: " << device_tensor->ToString();
    return nullptr;
  }
  return host_tensor;
}
}  // namespace

// --- Ascend memory-pool free-function helpers (declared in sliding_window_internal.h) ---

size_t AscendPoolIdleSize() {
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  return ctx->device_res_manager_->GetTotalIdleMemStatistics();
}

size_t AscendPoolExpandableSize() {
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  // Lazily initialize max_mem on the first successful call (pool isn't ready at startup).
  static size_t max_mem = 0;
  if (max_mem == 0) {
    auto *pool = ctx->device_res_manager_->GetMemoryPool();
    if (pool != nullptr) {
      max_mem = static_cast<size_t>(pool->total_mem_size());
      MS_LOG(INFO) << "Initialized max_mem from memory pool: " << (max_mem / 1024 / 1024) << " MB";
    }
  }
  size_t total = ctx->device_res_manager_->GetTotalMemStatistics();
  return (max_mem <= total) ? 0 : (max_mem - total);
}

size_t AscendPoolEventHeldSize() {
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  auto *pool = ctx->device_res_manager_->GetMemoryPool();
  return (pool == nullptr) ? 0 : pool->TotalUsedByEventMemStatistics();
}

void SyncAscendPoolEvents() {
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  auto *pool = ctx->device_res_manager_->GetMemoryPool();
  if (pool != nullptr) {
    pool->SyncAllEvents();
  }
}

bool IsNotEventUsedMemory(device::DeviceAddress *address) {
  if (address == nullptr) {
    return true;
  }
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  return ctx->device_res_manager_->IsNotEventUsedMemory(address);
}

void PrintAscendMemStatistics(const std::string &location) {
  auto *ctx = GetAscendDeviceContext();
  MS_EXCEPTION_IF_NULL(ctx);
  MS_EXCEPTION_IF_NULL(ctx->device_res_manager_);
  const auto &res = ctx->device_res_manager_;
  MS_LOG(INFO) << "MemoryStatistics [" << location
                                   << "]: total=" << (res->GetTotalMemStatistics() / 1024.0 / 1024.0)
                                   << " MB, used=" << (res->GetTotalUsedMemStatistics() / 1024.0 / 1024.0)
                                   << " MB, idle=" << (res->GetTotalIdleMemStatistics() / 1024.0 / 1024.0)
                                   << " MB, eager_free=" << (res->GetTotalEagerFreeMemStatistics() / 1024.0 / 1024.0)
                                   << " MB, used_peak=" << (res->GetUsedMemPeakStatistics() / 1024.0 / 1024.0)
                                   << " MB, reserved_peak=" << (res->GetReservedMemPeakStatistics() / 1024.0 / 1024.0)
                                   << " MB";
  auto block_counts = res->GetBlockCountsStatistics();
  for (const auto &[block_type, count] : block_counts) {
    MS_LOG(INFO) << "  Block " << block_type << ": " << count;
  }
}

DeviceSpaceManager::DeviceSpaceManager(SourceGroupManager *source_group_mgr) : source_group_mgr_(source_group_mgr) {
  MS_EXCEPTION_IF_NULL(source_group_mgr_);
}

size_t DeviceSpaceManager::TryOffloadGroup(SourceGroup *group) {
  if (group == nullptr || group->status != ResidencyStatus::kOnDevice || !group->offloadable) {
    return 0;
  }

  auto source = group->source.lock();
  if (!source) {
    group->offloadable = false;
    return 0;
  }
  if (!CanOffloadTensor(source)) {
    group->offloadable = false;
    return 0;
  }
  auto addr = source->device_address();
  if (addr == nullptr || addr->GetDevicePtr() == nullptr || !IsNotEventUsedMemory(addr.get())) {
    return 0;
  }

  auto host_tensor = SwapTensorToHost(source);
  if (host_tensor == nullptr) {
    MS_LOG(INFO) << "Offload failed: source_id=" << group->source_id;
    return 0;
  }

  group->host_copy = host_tensor;
  source_group_mgr_->UpdateAddressMapping(group, source->device_address());
  group->status = ResidencyStatus::kOnHost;
  group->aligned_size = AlignToPoolSize(source->Size());
  source->device_address()->ClearDeviceMemory();

  MS_LOG(INFO) << "Offloaded source group: id=" << group->source_id
                                   << " (" << (source->Size() / 1024 / 1024) << " MB)";
  return group->aligned_size;
}

void DeviceSpaceManager::FinalizeOffload() {
  // Wait for async ClearDeviceMemory tasks to complete so freed memory
  // is returned to the pool.
  runtime::Pipeline::Get().WaitAll();
  // Other runtime copies may have recorded cross-stream events on memory buffers.
  // SyncAllEvents resolves pending events and releases event-held memory back to idle.
  SyncAscendPoolEvents();
}

bool DeviceSpaceManager::ReloadDeviceAddress(TensorId source_id, const device::DeviceAddressPtr &device_addr,
                                             const TensorPtr &host_copy) {
  MS_EXCEPTION_IF_NULL(host_copy);
  MS_EXCEPTION_IF_NULL(device_addr);

  if (device_addr->GetDevicePtr() != nullptr) {
    MS_LOG(ERROR) << "Skip reload for offloaded tensor id=" << source_id
                  << ", shared DeviceAddress already has device ptr.";
    return true;
  }

  TensorPtr temp_device_tensor = SwapTensorToDevice(host_copy, device_addr->stream_id());
  if (temp_device_tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to reload offloaded tensor to device. Tensor id: " << source_id;
    return false;
  }
  auto new_device_addr = temp_device_tensor->device_address();
  MS_EXCEPTION_IF_NULL(new_device_addr);

  device_addr->SetDevicePtr(new_device_addr->GetDevicePtr());
  new_device_addr->set_from_mem_pool(false);  // prevent double-free
  return true;
}

bool DeviceSpaceManager::ReloadGroup(SourceGroup *group, const TensorPtr &addr_tensor) {
  if (group == nullptr || group->status != ResidencyStatus::kOnHost) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(group->host_copy);

  device::DeviceAddressPtr device_addr = nullptr;
  if (addr_tensor != nullptr) {
    device_addr = addr_tensor->device_address();
  }
  if (device_addr == nullptr) {
    if (auto source = group->source.lock(); source != nullptr) {
      device_addr = source->device_address();
    }
  }
  if (device_addr == nullptr) {
    device_addr = group->shared_addr.lock();
  }
  if (device_addr == nullptr) {
    MS_LOG(INFO) << "Skip reload for source group id=" << group->source_id
                                       << ", shared DeviceAddress expired.";
    group->host_copy = nullptr;
    return false;
  }

  if (!ReloadDeviceAddress(group->source_id, device_addr, group->host_copy)) {
    return false;
  }
  source_group_mgr_->UpdateAddressMapping(group, device_addr);
  group->host_copy = nullptr;
  group->status = ResidencyStatus::kOnDevice;
  return true;
}

bool DeviceSpaceManager::EnsureDeviceSpaceEnough(size_t expected_size,
                                                 const std::unordered_set<TensorId> &protected_source_ids) {
  source_group_mgr_->PurgeExpired();

  // Optimistic event sync — cheap no-op when nothing is pending, and frees
  // event-held mem_bufs so they become eligible for reuse.
  runtime::Pipeline::Get().WaitAll();
  SyncAscendPoolEvents();

  size_t available = AscendPoolIdleSize() + AscendPoolExpandableSize();
  if (available >= expected_size) {
    return true;
  }
  size_t size_needed = expected_size - available;

  MS_LOG(INFO) << "EnsureDeviceSpaceEnough: need "
                                 << (size_needed / 1024 / 1024) << " MB, protected=" << protected_source_ids.size();

  size_t freed = 0;
  size_t offloaded = 0;
  for (auto source_id : source_group_mgr_->GetOffloadCandidatesLRU()) {
    if (freed >= size_needed) break;
    if (protected_source_ids.count(source_id) > 0) continue;
    auto *group = source_group_mgr_->GetGroup(source_id);
    auto bytes = TryOffloadGroup(group);
    if (bytes > 0) {
      freed += bytes;
      ++offloaded;
    }
  }

  if (offloaded > 0) {
    FinalizeOffload();
    MS_LOG(INFO) << "EnsureDeviceSpaceEnough done: offloaded=" << offloaded
                                   << ", freed=" << (freed / 1024 / 1024)
                                   << " MB, source_groups=" << source_group_mgr_->Size();
  }
  size_t available_after = AscendPoolIdleSize() + AscendPoolExpandableSize();
  if (available_after < expected_size) {
    MS_LOG(INFO) << "EnsureDeviceSpaceEnough could not free enough memory: need="
                                      << (size_needed / 1024 / 1024) << " MB, freed="
                                      << (freed / 1024 / 1024) << " MB, available_after="
                                      << (available_after / 1024 / 1024) << " MB";
    return false;
  }
  return true;
}

bool DeviceSpaceManager::EnsureInputsOnDevice(const std::vector<TensorPtr> &inputs) {
  MS_LOG(INFO) << "EnsureInputsOnDevice: inputs=" << inputs.size();

  size_t reload_count = 0;
  size_t on_device_count = 0;
  bool success = true;

  // Track already-reloaded source IDs to avoid reloading the same source twice
  std::unordered_set<TensorId> reloaded_sources;
  const auto protected_source_ids = source_group_mgr_->BuildProtectedSourceIdSet(inputs);

  for (const auto &tensor : inputs) {
    if (tensor == nullptr) {
      continue;
    }

    TensorId tensor_id = tensor->id();
    TensorId lookup_id = source_group_mgr_->ResolveSourceId(tensor);

    // Skip if we already reloaded this source for a previous input
    if (reloaded_sources.count(lookup_id) > 0) {
      on_device_count++;
      continue;
    }

    auto *group = source_group_mgr_->GetGroup(lookup_id);
    if (group != nullptr) {
      source_group_mgr_->RegisterAlias(tensor, group);
    }
    if (group == nullptr || group->status == ResidencyStatus::kOnDevice) {
      auto addr = tensor->device_address();
      if (addr == nullptr || addr->GetDevicePtr() == nullptr) {
        MS_LOG(INFO) << "  No device memory and not offloaded: id=" << tensor_id
                                           << (lookup_id != tensor_id
                                                 ? " (source=" + std::to_string(lookup_id) + ")"
                                                 : "");
      }
      on_device_count++;
      continue;
    }

    MS_EXCEPTION_IF_NULL(group->host_copy);
    auto reload_size = AlignToPoolSize(group->host_copy->Size());
    if (reload_size > 0) {
      success = EnsureDeviceSpaceEnough(reload_size, protected_source_ids) && success;
    }

    MS_LOG(INFO) << "  Reloading tensor [" << reload_count << "]: id=" << lookup_id
                                   << (lookup_id != tensor_id
                                         ? " (triggered by view " + std::to_string(tensor_id) + ")"
                                         : "")
                                   << ", size=" << (reload_size / 1024 / 1024) << " MB";

    success = ReloadGroup(group, tensor) && success;
    reloaded_sources.insert(lookup_id);
    reload_count++;
  }

  MS_LOG(INFO) << "EnsureInputsOnDevice done: reloaded=" << reload_count
                                   << ", on_device=" << on_device_count
                                   << ", source_groups=" << source_group_mgr_->Size();
  return success;
}

void DeviceSpaceManager::PurgeExpiredOffloaded() {
  source_group_mgr_->PurgeExpired();
}

bool DeviceSpaceManager::EnsureAllOffloadedOnDevice() {
  MS_LOG(INFO) << "EnsureAllOffloadedOnDevice: source_groups=" << source_group_mgr_->Size();
  size_t reload_count = 0;
  size_t skipped_count = 0;
  bool success = true;

  // Process in arbitrary order (no LRU priority needed for bulk reload)
  for (auto source_id : source_group_mgr_->GetAllSourceIds()) {
    auto *group = source_group_mgr_->GetGroup(source_id);
    if (group == nullptr || group->status != ResidencyStatus::kOnHost) {
      continue;
    }

    auto source_tensor = group->source.lock();
    auto shared_device_addr = group->shared_addr.lock();

    if (source_tensor == nullptr && shared_device_addr == nullptr) {
      MS_LOG(INFO) << "DeviceAddress expired for offloaded source group id=" << source_id
                                         << ", remove dangling host copy.";
      group->host_copy = nullptr;
      group->status = ResidencyStatus::kOnDevice;
      ++skipped_count;
      continue;
    }

    MS_EXCEPTION_IF_NULL(group->host_copy);
    MS_LOG(INFO) << "  Reloading offloaded source group: id=" << source_id
                                   << " size=" << (group->host_copy->Size() / 1024 / 1024) << " MB"
                                   << (source_tensor == nullptr ? " via surviving DeviceAddress" : "");

    success = ReloadGroup(group, source_tensor) && success;
    ++reload_count;
  }

  FinalizeOffload();

  MS_LOG(INFO) << "EnsureAllOffloadedOnDevice done: reloaded=" << reload_count
                                   << ", skipped=" << skipped_count
                                   << ", source_groups=" << source_group_mgr_->Size();
  return success;
}


}  // namespace sliding_window
}  // namespace utils
}  // namespace mindspore
