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

#include <set>
#include <map>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include <mutex>
#include <vector>
#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace plugin_loader {
class PluginLoader {
 public:
  static bool LoadDynamicLib(const std::string &plugin_file, std::map<std::string, void *> *all_handles,
                             std::stringstream *err_msg);
  static void CloseDynamicLib(const std::string &dl_name, void *handle);
  static bool GetPluginPath(std::string *file_path);

 private:
  static std::string GetDynamicLibName(const std::string &plugin_file);
};
}  // namespace plugin_loader

namespace device {
using DeviceContextCreator = std::function<std::shared_ptr<DeviceContext>(const DeviceContextKey &)>;

const DeviceContext *FetchRealDeviceContext(const CNodePtr &node, const DeviceContext *device_context);

class BACKEND_EXPORT DeviceContextManager {
 public:
  static DeviceContextManager &GetInstance();
  void Register(const std::string &device_name, DeviceContextCreator &&device_context_creator);
  DeviceContext *GetOrCreateDeviceContext(const DeviceContextKey &device_context_key, string jit_level = "");
  void UpdateDeviceContextKey(const DeviceContextKey &old_key, const DeviceContextKey &new_key);
  void ClearDeviceContexts();
  void WaitTaskFinishOnDevice() const;
  void UnloadPlugin();
  std::string GetErrorMsg() const;
  void BindDeviceCtx() const;

 private:
  DeviceContextManager() = default;
  ~DeviceContextManager() = default;
  DISABLE_COPY_AND_ASSIGN(DeviceContextManager);
  void LoadPlugin();
  void SelectGpuPlugin(const std::string &cuda_home, const std::set<std::string> &file_names);

  std::map<std::string, void *> plugin_maps_;
  bool load_init_;
  std::string plugin_path_;

  // The string converted from DeviceContextKey -> DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> device_contexts_;
  // The name of device -> DeviceContextCreator.
  std::map<std::string, DeviceContextCreator> device_context_creators_;
  // record error message of dlopen, print when create device_context failed.
  std::stringstream dlopen_error_msg_;
};

class BACKEND_EXPORT DeviceContextRegister {
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
