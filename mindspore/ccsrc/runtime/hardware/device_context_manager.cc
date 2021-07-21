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

#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
void DeviceContextManager::Register(const std::string &device_name, DeviceContextCreator &&device_context_creator) {
  if (device_context_creators_.find(device_name) == device_context_creators_.end()) {
    (void)device_context_creators_.emplace(device_name, device_context_creator);
  }
}

void DeviceContextManager::ClearDeviceContexts() {
  for (auto &iter : device_contexts_) {
    MS_LOG(INFO) << "Release device " << iter.first;
    MS_EXCEPTION_IF_NULL(iter.second);
    iter.second->Destroy();
  }
  device_contexts_.clear();
}

DeviceContext *DeviceContextManager::GetOrCreateDeviceContext(const DeviceContextKey &device_context_key) {
  std::string device_context_key_str = device_context_key.ToString();

  auto device_context_iter = device_contexts_.find(device_context_key_str);
  if (device_context_iter != device_contexts_.end()) {
    return device_context_iter->second.get();
  }

  std::shared_ptr<DeviceContext> device_context;
  auto creator_iter = device_context_creators_.find(device_context_key.device_name_);
  if (creator_iter != device_context_creators_.end()) {
    device_context = (creator_iter->second)(device_context_key);
    MS_EXCEPTION_IF_NULL(device_context);
    device_contexts_[device_context_key_str] = device_context;
  } else {
    MS_LOG(EXCEPTION) << "There is no device context creator for " << device_context_key.device_name_
                      << " with device id " << device_context_key.device_id_;
  }
  return device_context.get();
}

void DeviceContextManager::UpdateDeviceContextKey(const DeviceContextKey &old_key, const DeviceContextKey &new_key) {
  std::string old_key_str = old_key.ToString();
  std::string new_key_str = new_key.ToString();

  auto handle = device_contexts_.extract(old_key_str);
  if (handle.empty()) {
    MS_LOG(EXCEPTION) << "Can not find device context for: " << old_key_str;
  }

  handle.key() = new_key_str;
  (void)device_contexts_.insert(std::move(handle));
}
}  // namespace device
}  // namespace mindspore
