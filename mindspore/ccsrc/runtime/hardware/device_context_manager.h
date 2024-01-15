/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <any>
#include <map>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include <mutex>
#include <vector>
#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"
#include "include/common/pybind_api/api_register.h"

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

// This callback registers stateless functions to _c_expression. It is set by different device contexts.
using RegisterStatelessFuncCb = std::function<void(py::module *m)>;

const DeviceContext *FetchRealDeviceContext(const AnfNodePtr &node, const DeviceContext *device_context);

class BACKEND_EXPORT DeviceContextManager {
 public:
  ~DeviceContextManager() = default;
  static DeviceContextManager &GetInstance();
  void Register(const std::string &device_name, DeviceContextCreator &&device_context_creator);
  DeviceContext *GetOrCreateDeviceContext(const DeviceContextKey &device_context_key);
  // Return the device context of the specified device target.
  // The difference between this method and 'GetOrCreateDeviceContext' is this method only query device context by
  // device target(without device id) since MindSpore only supports 'single process, single device'.
  DeviceContextPtr GetDeviceContext(const std::string &device_target);
  void UpdateDeviceContextKey(const DeviceContextKey &old_key, const DeviceContextKey &new_key);
  void ClearDeviceContexts();
  void WaitTaskFinishOnDevice() const;
  void UnloadPlugin();
  std::string GetErrorMsg() const;
  void BindDeviceCtx() const;

  // For different device backends, some methods are stateless. They have to be registered to `DeviceContextManager`.
  void SetRegisterDeviceStatelessFuncCb(const std::string &backend, const RegisterStatelessFuncCb &register_func_cb);
  void RegisterDeviceStatelessFunc(py::module *m);

 private:
  DeviceContextManager() = default;
  DISABLE_COPY_AND_ASSIGN(DeviceContextManager);
  void LoadPlugin();
  void SelectGpuPlugin(const std::string &cuda_home, const std::set<std::string> &file_names);

  std::map<std::string, void *> plugin_maps_;
  bool load_init_;
  std::string plugin_path_;

  // The string converted from DeviceContextKey -> DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> device_contexts_;
  // The name of device -> vector of DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> backend_to_device_context_;
  // The name of device -> DeviceContextCreator.
  std::map<std::string, DeviceContextCreator> device_context_creators_;
  // record error message of dlopen, print when create device_context failed.
  std::stringstream dlopen_error_msg_;

  // Backend name->register stateless functions callback.
  std::map<std::string, RegisterStatelessFuncCb> register_func_cbs_;
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

class BACKEND_EXPORT StatelessFuncCbRegister {
 public:
  StatelessFuncCbRegister(const std::string &device_name, const RegisterStatelessFuncCb &func) {
    DeviceContextManager::GetInstance().SetRegisterDeviceStatelessFuncCb(device_name, func);
  }
  ~StatelessFuncCbRegister() = default;
};

#define REGISTER_DEV_STATELESS_FUNC_CB(DEVICE_NAME, FUNC_OBJ) \
  static const StatelessFuncCbRegister g_##DEVICE_NAME##_stateless_func_cb_reg(DEVICE_NAME, FUNC_OBJ)
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_CONTEXT_MANAGER_H_
