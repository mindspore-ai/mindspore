/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "pybind_api/backend/graph/custom_pass_py.h"
#include <string>
#include "include/backend/common/custom_pass/custom_pass_plugin_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace graph {

void RegCustomPass(py::module *m) {
  // Register custom pass plugin from Python
  (void)m->def(
    "register_custom_pass",
    [](const std::string &pass_name, const std::string &plugin_so_path, const std::string &device,
       const std::string &stage) -> bool {
      using mindspore::opt::CustomPassPluginManager;
      if (plugin_so_path.empty()) {
        MS_LOG(ERROR) << "Plugin path is empty";
        return false;
      }
      if (device.empty()) {
        MS_LOG(ERROR) << "Device parameter is empty";
        return false;
      }
      // Load plugin shared library with device and stage specification
      bool loaded = CustomPassPluginManager::GetInstance().LoadPlugin(plugin_so_path, pass_name, device, stage);
      if (!loaded) {
        MS_LOG(ERROR) << "Failed to load custom pass plugin from: " << plugin_so_path << " for device: " << device;
        return false;
      }
      // Log successful plugin loading
      if (!pass_name.empty()) {
        MS_LOG(INFO) << "Successfully loaded custom pass plugin: " << pass_name << " for device: " << device;
      }
      return true;
    },
    py::arg("pass_name"), py::arg("plugin_so_path"), py::arg("device"), py::arg("stage"),
    "Register a custom optimization pass by loading plugin shared library for specific device");
}

}  // namespace graph
}  // namespace mindspore
