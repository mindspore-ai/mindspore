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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_PLUGIN_LOADER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_PLUGIN_LOADER_H_

#include <map>
#include <string>
#include <utility>

#include "minddata/dataset/plugin/include/shared_include.h"
#include "minddata/dataset/util/status.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore {
namespace dataset {

// This class manages all MindData's plugins. It serves as the singleton that owns all plugins and bridge the gap
// between C++ RAII and C style functions
class PluginLoader {
 public:
  /// \brief Singleton getter,
  /// \return pointer to PluginLoader
  static PluginLoader *GetInstance() noexcept;

  PluginLoader() = default;

  /// \brief destructor, will call unload internally to unload all plugins managed by PluginLoader
  ~PluginLoader();

  /// \brief load an shared object (.so file) via dlopen() and return the ptr to the loaded file (singleton_plugin).
  /// \param[in] filename the full path to .so file
  /// \param[out] singleton_plugin pointer to the loaded file
  /// \return status code
  Status LoadPlugin(const std::string &filename, plugin::PluginManagerBase **singleton_plugin);

 private:
  /// \brief Unload so file, internally will call dlclose() and delete its handle.
  /// \param[in] filename, the full path to .so file
  /// \return status code
  Status UnloadPlugin(const std::string &filename);

  std::map<std::string, std::pair<plugin::PluginManagerBase *, void *>>
    plugins_;  // key: path, val: plugin, dlopen handle
};

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_PLUGIN_PLUGIN_LOADER_H_
