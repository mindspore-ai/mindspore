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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CUSTOM_PASS_CUSTOM_PASS_EXECUTOR_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CUSTOM_PASS_CUSTOM_PASS_EXECUTOR_H_

#include <memory>
#include <string>
#include <utility>
#include "include/utils/custom_pass/custom_pass_plugin_manager.h"
#include "utils/ordered_map.h"

namespace mindspore {
namespace pipeline {
class Resource;
using ResourcePtr = std::shared_ptr<Resource>;
using ActionItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;
}  // namespace pipeline

namespace opt {

class CustomPassExecutor {
 public:
  // Insert frontend custom passes
  static void InsertFrontendCustomPassesToJitPipeline(
    OrderedMap<std::string, std::function<bool(pipeline::ResourcePtr)>> &jit_passes);
  static void InsertFrontendCustomPassesToVmPipeline(std::vector<pipeline::ActionItem> &actions);

 private:
  CustomPassExecutor() = delete;
  ~CustomPassExecutor() = delete;

  // Create frontend action wrapper for custom pass
  static std::function<bool(pipeline::ResourcePtr)> CreateFrontendPassAction(
    const std::shared_ptr<CustomPassPlugin> &plugin, const std::string &pass_name);

  // Process frontend passes with a callback function
  template <typename Func>
  static void ProcessFrontendPasses(Func callback) {
    auto &plugin_manager = CustomPassPluginManager::GetInstance();
    const auto &pass_execs = plugin_manager.GetPassExecutionOrder();
    const auto &plugins = plugin_manager.GetPlugins();
    for (const auto &pass_exec : pass_execs) {
      // Check if this is a frontend custom pass
      if (pass_exec.pass_type != CustomPassType::kFullGraph) {
        MS_LOG(DEBUG) << "Pass '" << pass_exec.pass_name << "' has pass_type '" << pass_exec.pass_type
                      << "' Skipping frontend optimizer registration.";
        continue;
      }

      // Find plugin by name
      auto plugin_iter = plugins.find(pass_exec.plugin_name);
      if (plugin_iter == plugins.end() || !plugin_iter->second || !plugin_iter->second->plugin) {
        MS_LOG(WARNING) << "Plugin '" << pass_exec.plugin_name << "' for frontend pass '" << pass_exec.pass_name
                        << "' not found or invalid";
        continue;
      }

      MS_EXCEPTION_IF_NULL(plugin_iter->second);
      const auto &plugin = plugin_iter->second->plugin;

      // Create action wrapper
      auto action = CreateFrontendPassAction(plugin, pass_exec.pass_name);

      // Call callback with pass information
      std::string pass_key = "custom_frontend_pass_" + pass_exec.pass_name;
      callback(pass_key, action);
      MS_LOG(INFO) << "Process '" << pass_key << "' from plugin '" << plugin->GetPluginName();
    }
  }
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CUSTOM_PASS_CUSTOM_PASS_EXECUTOR_H_
