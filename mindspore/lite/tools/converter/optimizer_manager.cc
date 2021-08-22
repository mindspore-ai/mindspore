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

#include "tools/converter/optimizer_manager.h"
#include <string>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "tools/converter/registry/pass_content.h"

namespace mindspore {
namespace opt {
bool RunOptimizerPass(const FuncGraphPtr &func_graph, std::vector<std::string> pass_names) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is nullptr.";
    return false;
  }
  auto &passes_info = PassStoreRoomInfo();
  for (auto &name : pass_names) {
    if (passes_info.find(name) == passes_info.end()) {
      MS_LOG(ERROR) << "cannot find required pass.";
      return false;
    }
    if (!passes_info[name]->Run(func_graph)) {
      MS_LOG(ERROR) << "run pass failed, pass name is " << name;
      return false;
    }
  }
  return true;
}

bool RunExternalPass(const FuncGraphPtr &func_graph, PassPosition position) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is nullptr.";
    return false;
  }
  auto &external_assigned = ExternalAssignedPassesInfo();
  if (external_assigned.find(position) == external_assigned.end()) {
    MS_LOG(DEBUG) << "there is no external pass in current position, position is " << position;
    return true;
  }
  auto &passes_info = PassStoreRoomInfo();
  for (auto &name : external_assigned[position]) {
    if (passes_info.find(name) == passes_info.end()) {
      MS_LOG(ERROR) << "cannot find required pass.";
      return false;
    }
    if (!passes_info[name]->Run(func_graph)) {
      MS_LOG(ERROR) << "run pass failed, pass name is " << name;
      return false;
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
