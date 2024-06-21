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

#include "runtime/hardware/device_context.h"
#include "backend/common/optimizer/common_backend_optimization.h"

namespace mindspore {
namespace device {
void *DeviceResManager::AllocateOffloadMemory(size_t size) const { return offloaded_mem_pool_->MallocHost(size); }

void DeviceResManager::FreeOffloadMemory(void *ptr) const { offloaded_mem_pool_->FreeHost(ptr); }

bool DeviceResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }
  auto device_ptr = AllocateMemory(address->GetSize(), stream_id);
  if (!device_ptr) {
    MS_LOG(WARNING) << "Allocate memory failed for size: " << address->GetSize();
    return false;
  }
  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  return true;
}

void DeviceResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Device ptr is null in device address:" << address << " to release!";
  }

  if (!address->from_mem_pool()) {
    MS_LOG(DEBUG) << "device address:" << address << " ptr:" << address->GetMutablePtr() << " not from pool";
    return;
  }
  MS_LOG(DEBUG) << "Free memory from device address:" << address << " ptr:" << address->GetMutablePtr();
  FreeMemory(address->GetMutablePtr());
  address->set_ptr(nullptr);
}

bool DeviceResManager::DestroyEvent(const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (!event->DestroyEvent()) {
    MS_LOG(ERROR) << "DestroyEvent failed.";
    return false;
  }

  std::lock_guard<std::mutex> lock(device_events_mutex_);
  const auto &iter = std::find(device_events_.begin(), device_events_.end(), event);
  if (iter == device_events_.end()) {
    MS_LOG(ERROR) << "Can't find specified device event.";
    return false;
  }
  (void)device_events_.erase(iter);
  return true;
}

bool DeviceResManager::DestroyAllEvents() {
  DeviceEventPtrList device_events_inner;
  {
    // Reduce the scopt to prevent deadlock.
    std::lock_guard<std::mutex> lock(device_events_mutex_);
    device_events_inner = device_events_;
    device_events_.clear();
  }
  (void)std::for_each(device_events_inner.begin(), device_events_inner.end(), [this](const auto &event) {
    MS_EXCEPTION_IF_NULL(event);
    if (!event->DestroyEvent()) {
      MS_LOG(ERROR) << "DestroyEvent failed.";
    }
  });
  device_events_.clear();
  return true;
}

void KernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const { opt::CommonUnifyMindIR(graph); }
}  // namespace device
}  // namespace mindspore
