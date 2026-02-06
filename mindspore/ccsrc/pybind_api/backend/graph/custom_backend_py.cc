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

#include "pybind_api/backend/graph/custom_backend_py.h"
#include <string>
#include "include/backend/backend_manager/backend_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace graph {

void RegCustomBackend(py::module *m) {
  // Register custom pass plugin from Python
  (void)m->def(
    "register_custom_backend",
    [](const std::string &backend_name, const std::string &backend_path) -> bool {
      using mindspore::backend::BackendManager;
      if (backend_path.empty()) {
        MS_LOG(ERROR) << "Plugin path is empty";
        return false;
      }
      // Load plugin shared library
      bool loaded = BackendManager::GetInstance().LoadBackend(backend_name, backend_path);
      if (!loaded) {
        MS_LOG(ERROR) << "Failed to load custom backend plugin from: " << backend_path
                      << " for backend: " << backend_name << ", please check the plugin path and name is valid";
        return false;
      }
      MS_LOG(INFO) << "Successfully load custom backend plugin from: " << backend_path
                   << " for backend: " << backend_name;
      return true;
    },
    py::arg("backend_name"), py::arg("plugin_so_path"), "Register a custom backend by loading plugin shared library.");
}

}  // namespace graph
}  // namespace mindspore
