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

#include "frontend/optimizer/custom_pass/custom_pass_executor.h"

#include "include/backend/common/pass_manager/pass.h"
#include "include/frontend/jit/ps/resource_interface.h"

namespace mindspore {
namespace opt {

std::function<bool(pipeline::ResourcePtr)> CustomPassExecutor::CreateFrontendPassAction(
  const std::shared_ptr<CustomPassPlugin> &plugin, const std::string &pass_name) {
  return [plugin, pass_name](const pipeline::ResourcePtr &resource) -> bool {
    MS_EXCEPTION_IF_NULL(resource);
    auto func_graph = resource->func_graph();
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "func_graph is null when executing custom frontend pass: " << pass_name;
      return false;
    }

    // Create the custom pass
    auto pass = plugin->CreatePass(pass_name);
    if (pass == nullptr) {
      MS_LOG(ERROR) << "Failed to create frontend pass '" << pass_name << "' from plugin '" << plugin->GetPluginName()
                    << "'";
      return false;
    }

    // Execute the pass on the function graph
    MS_LOG(INFO) << "Executing custom frontend pass: " << pass_name;
    bool changed = pass->Run(func_graph);
    MS_LOG(INFO) << "Custom frontend pass '" << pass_name << "' execution completed, changed=" << changed;
    return true;
  };
}

void CustomPassExecutor::InsertFrontendCustomPassesToJitPipeline(
  OrderedMap<std::string, std::function<bool(pipeline::ResourcePtr)>> &jit_passes) {
  ProcessFrontendPasses(
    [&jit_passes](const std::string &pass_key, const std::function<bool(pipeline::ResourcePtr)> &action) {
      jit_passes.emplace(pass_key, action);
    });
}

void CustomPassExecutor::InsertFrontendCustomPassesToVmPipeline(std::vector<pipeline::ActionItem> &actions) {
  ProcessFrontendPasses(
    [&actions](const std::string &pass_key, const std::function<bool(pipeline::ResourcePtr)> &action) {
      actions.emplace_back(std::make_pair(pass_key, action));
    });
}

}  // namespace opt
}  // namespace mindspore
