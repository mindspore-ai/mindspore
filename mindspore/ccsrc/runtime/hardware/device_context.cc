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

namespace mindspore {
namespace device {
bool DeviceContext::CreateStream(size_t *stream_id) {
  MS_EXCEPTION_IF_NULL(stream_id);
  std::lock_guard<std::mutex> locker(stream_mutex_);
  void *stream = nullptr;
  if (!CreateStream(&stream)) {
    MS_LOG(ERROR) << "Create stream failed.";
    return false;
  }
  MS_EXCEPTION_IF_NULL(stream);

  // Because the default value of the stream_id parameter in the current SyncStream interface is 0, the stream_id is not
  // used when sync computation stream currently, and the current stream creation method is not unified. In order not to
  // affect the synchronization of ordinary computation streams, the stream id here starts from 1. After unifying
  // stream creation method in the future, this restriction can be lifted.
  static size_t stream_id_cnt = 1;
  stream_ids_[stream_id_cnt] = stream;
  *stream_id = stream_id_cnt++;
  return true;
}

bool DeviceContext::DestroyAllStreams() {
  for (auto &item : stream_ids_) {
    if (item.second != nullptr) {
      if (!DestroyStream(item.second)) {
        MS_LOG(ERROR) << "Destroy stream failed";
        return false;
      }
      item.second = nullptr;
    }
  }

  stream_ids_.clear();
  return true;
}

void *DeviceContext::GetStream(size_t stream_id) const {
  auto iter = stream_ids_.find(stream_id);
  if (iter == stream_ids_.end()) {
    MS_LOG(ERROR) << "Can not find stream for stream id[" << stream_id << "]";
    return nullptr;
  }

  return iter->second;
}

bool DeviceContext::AllocateMemory(DeviceAddress *const &address, size_t size) const {
  MS_EXCEPTION_IF_NULL(address);
  auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
  if (device_name_in_address != device_context_key_.device_name_) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                      << ", type name in context:" << device_context_key_.device_name_;
  }

  if (address->GetPtr() != nullptr) {
    MS_LOG(EXCEPTION) << "Memory leak detected!";
  }

  auto device_ptr = AllocateMemory(size);
  if (!device_ptr) {
    return false;
  }
  address->set_ptr(device_ptr);
  address->SetSize(size);
  address->set_from_mem_pool(true);
  return true;
}

void DeviceContext::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Device ptr is null in device address to release!";
  }

  auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
  if (device_name_in_address != device_context_key_.device_name_) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                      << ", type name in context:" << device_context_key_.device_name_;
  }

  if (!address->from_mem_pool()) {
    return;
  }

  void *ptr = const_cast<void *>(address->GetPtr());
  FreeMemory(ptr);
  address->set_ptr(nullptr);
}
}  // namespace device
}  // namespace mindspore
