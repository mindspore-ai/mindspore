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

#include "include/utils/custom_pass/custom_pass_executor_base.h"

namespace mindspore {
namespace opt {

void CustomPassExecutorBase::ProcessPasses(
  const std::function<bool(const CustomPassPluginManager::PassExecution &)> &filter,
  const std::function<void(const std::string &pass_name, const std::shared_ptr<CustomPassPlugin> &plugin)> &handler) {
  MS_EXCEPTION_IF_NULL(filter);
  MS_EXCEPTION_IF_NULL(handler);
  auto &plugin_manager = CustomPassPluginManager::GetInstance();
  const auto &pass_execs = plugin_manager.GetPassExecutionOrder();
  const auto &plugins = plugin_manager.GetPlugins();

  for (const auto &pass_exec : pass_execs) {
    // Apply filter
    if (!filter(pass_exec)) {
      continue;
    }

    // Find plugin by name directly (more efficient than searching by pass name)
    auto plugin_iter = plugins.find(pass_exec.plugin_name);
    if (plugin_iter == plugins.end() || !plugin_iter->second || !plugin_iter->second->plugin) {
      MS_LOG(WARNING) << "Plugin '" << pass_exec.plugin_name << "' for pass '" << pass_exec.pass_name
                      << "' not found or invalid";
      continue;
    }

    const auto &plugin = plugin_iter->second->plugin;
    // Call handler with pass information
    handler(pass_exec.pass_name, plugin);
  }
}

}  // namespace opt
}  // namespace mindspore
