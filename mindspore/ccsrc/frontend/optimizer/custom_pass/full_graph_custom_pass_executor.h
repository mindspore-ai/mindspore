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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CUSTOM_PASS_FULL_GRAPH_CUSTOM_PASS_EXECUTOR_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CUSTOM_PASS_FULL_GRAPH_CUSTOM_PASS_EXECUTOR_H_

#include "utils/ordered_map.h"
#include "include/utils/custom_pass/custom_pass_executor_base.h"

namespace mindspore {
namespace pipeline {
class Resource;
using ResourcePtr = std::shared_ptr<Resource>;
using ActionItem = std::pair<std::string, std::function<bool(ResourcePtr)>>;
}  // namespace pipeline

namespace opt {

class FullGraphCustomPassExecutor : public CustomPassExecutorBase {
 public:
  // Insert fullgraph custom passes
  static void InsertFullGraphCustomPassesToJitPipeline(
    OrderedMap<std::string, std::function<bool(pipeline::ResourcePtr)>> &jit_passes);
  static void InsertFullGraphCustomPassesToVmPipeline(std::vector<pipeline::ActionItem> &actions);

 private:
  FullGraphCustomPassExecutor() = delete;
  ~FullGraphCustomPassExecutor() = delete;

  // Create frontend action wrapper for custom pass
  static std::function<bool(pipeline::ResourcePtr)> CreateFrontendPassAction(
    const std::shared_ptr<CustomPassPlugin> &plugin, const std::string &pass_name);

  // Process fullgraph passes with a callback function
  template <typename Func>
  static void ProcessFullGraphPasses(Func callback) {
    ProcessPasses(
      [](const CustomPassPluginManager::PassExecution &pass_exec) {
        // Check if this is a fullgraph custom pass
        if (pass_exec.pass_type != CustomPassType::kFullGraph) {
          MS_LOG(DEBUG) << "Pass '" << pass_exec.pass_name << "' has pass_type '" << pass_exec.pass_type
                        << "' Skipping frontend optimizer registration.";
          return false;
        }
        return true;
      },
      [callback](const std::string &pass_name, const std::shared_ptr<CustomPassPlugin> &plugin) {
        // Create action wrapper
        auto action = CreateFrontendPassAction(plugin, pass_name);

        // Call callback with pass information
        std::string pass_key = "custom_frontend_pass_" + pass_name;
        callback(pass_key, action);
        MS_LOG(INFO) << "Process '" << pass_key << "' from plugin '" << plugin->GetPluginName() << "'";
      });
  }
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_CUSTOM_PASS_FULL_GRAPH_CUSTOM_PASS_EXECUTOR_H_
