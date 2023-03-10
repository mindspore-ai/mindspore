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

#define USE_DEPRECATED_API
#include "tools/converter/optimizer_manager.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "src/common/log_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "include/registry/pass_base.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
std::map<std::string, opt::PassPtr> PassStorage::pass_storage_;
std::set<std::string> PassStorage::inaccessible_for_outer_;
bool RunOptimizerPass(const FuncGraphPtr &func_graph, const std::vector<std::string> &pass_names) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is nullptr.";
    return false;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
    MS_CHECK_TRUE_RET(manager != nullptr, false);
    std::set<FuncGraphPtr> all_func_graphs;
    GetAllFuncGraph(func_graph, &all_func_graphs);
    for (auto &graph : all_func_graphs) {
      manager->AddFuncGraph(graph);
    }
  }
  for (auto &pass_name : pass_names) {
    auto pass_outer = registry::PassRegistry::GetPassFromStoreRoom(pass_name);
    if (pass_outer != nullptr) {
      auto api_graph = api::MakeShared<api::FuncGraph>(func_graph);
      MS_CHECK_TRUE_RET(api_graph != nullptr, false);
      if (!pass_outer->Execute(api_graph)) {
        MS_LOG(WARNING) << "run pass failed, pass name is " << pass_name;
        return false;
      }
      continue;
    }
    auto pass_builtin = PassStorage::GetPassFromStorage(pass_name);
    if (pass_builtin == nullptr) {
      MS_LOG(ERROR) << "exited pass cannot be obtained, pass name is " << pass_name;
      return false;
    }
    if (!pass_builtin->Run(func_graph)) {
      MS_LOG(WARNING) << "run pass failed, pass name is " << pass_name;
      return false;
    }
  }
  return true;
}

bool RunExternalPass(const FuncGraphPtr &func_graph, registry::PassPosition position) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is nullptr.";
    return false;
  }
  auto schedule_task = registry::PassRegistry::GetOuterScheduleTask(position);
  for (const auto &pass_name : schedule_task) {
    if (!PassStorage::IsAccessibleForOuter(pass_name)) {
      MS_LOG(ERROR) << pass_name << " is an inaccessible pass for outer calling.";
      return false;
    }
  }
  if (!RunOptimizerPass(func_graph, schedule_task)) {
    MS_LOG(WARNING) << "run external scheduled task failed.";
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
