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
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
bool RunOptimizerPass(const FuncGraphPtr &func_graph, const std::vector<std::string> &pass_names) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is nullptr.";
    return false;
  }
  auto schedule_passes = registry::PassRegistry::GetPassFromStoreRoom(pass_names);
  if (schedule_passes.size() != pass_names.size()) {
    MS_LOG(ERROR) << "exited pass cannot be obtained.";
    return false;
  }
  int index = 0;
  for (auto &pass : schedule_passes) {
    CHECK_NULL_RETURN(pass);
    if (!pass->Run(func_graph)) {
      MS_LOG(WARNING) << "run pass failed, pass name is " << pass_names[index];
      return false;
    }
    ++index;
  }
  return true;
}

bool RunExternalPass(const FuncGraphPtr &func_graph, registry::PassPosition position) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func graph is nullptr.";
    return false;
  }
  auto schedule_task = registry::PassRegistry::GetOuterScheduleTask(position);
  if (!RunOptimizerPass(func_graph, schedule_task)) {
    MS_LOG(ERROR) << "run external scheduled task failed.";
    return false;
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
