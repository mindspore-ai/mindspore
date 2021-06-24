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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_MANAGER_H_

#include <map>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include <mutex>
#include <vector>
#include "runtime/hardware/device_context.h"
namespace mindspore {
namespace device {
using DeviceContextCreator = std::function<std::shared_ptr<DeviceContext>(const DeviceContextKey &)>;

class DeviceContextManager {
 public:
  static DeviceContextManager &GetInstance() {
    static DeviceContextManager instance;
    return instance;
  }
  void Register(const std::string &device_name, DeviceContextCreator &&device_context_creator);
  DeviceContext *GetOrCreateDeviceContext(const DeviceContextKey &device_context_key);
  void UpdateDeviceContextKey(const DeviceContextKey &old_key, const DeviceContextKey &new_key);
  void ClearDeviceContexts();

 private:
  DeviceContextManager() = default;
  ~DeviceContextManager() = default;
  DISABLE_COPY_AND_ASSIGN(DeviceContextManager);

  // The string converted from DeviceContextKey -> DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> device_contexts_;
  // The name of device -> DeviceContextCreator.
  std::map<std::string, DeviceContextCreator> device_context_creators_;
};

class DeviceContextRegister {
 public:
  DeviceContextRegister(const std::string &device_name, DeviceContextCreator &&runtime_creator) {
    DeviceContextManager::GetInstance().Register(device_name, std::move(runtime_creator));
  }
  ~DeviceContextRegister() = default;
};

#define MS_REGISTER_DEVICE(DEVICE_NAME, DEVICE_CONTEXT_CLASS)            \
  static const DeviceContextRegister g_device_##DEVICE_NAME##_reg(       \
    DEVICE_NAME, [](const DeviceContextKey &device_context_key) {        \
      return std::make_shared<DEVICE_CONTEXT_CLASS>(device_context_key); \
    });
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_MANAGER_H_
