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

#include "include/registry/pass_registry.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace registry {
namespace {
std::map<std::string, opt::PassPtr> pass_store_room;
std::map<registry::PassPosition, std::vector<std::string>> external_assigned_passes;
std::mutex pass_mutex;
void RegPass(const std::string &pass_name, const opt::PassPtr &pass) {
  if (pass == nullptr) {
    MS_LOG(ERROR) << "pass is nullptr.";
    return;
  }
  std::unique_lock<std::mutex> lock(pass_mutex);
  pass_store_room[pass_name] = pass;
}
}  // namespace

PassRegistry::PassRegistry(const std::string &pass_name, const opt::PassPtr &pass) { RegPass(pass_name, pass); }

PassRegistry::PassRegistry(PassPosition position, const std::vector<std::string> &names) {
  std::unique_lock<std::mutex> lock(pass_mutex);
  external_assigned_passes[position] = names;
}

std::vector<std::string> PassRegistry::GetOuterScheduleTask(PassPosition position) {
  return external_assigned_passes[position];
}

std::vector<opt::PassPtr> PassRegistry::GetPassFromStoreRoom(const std::vector<std::string> &pass_names) {
  std::vector<opt::PassPtr> schedule_passes;
  for (auto &name : pass_names) {
    auto iter = pass_store_room.find(name);
    if (iter == pass_store_room.end()) {
      continue;
    }
    MS_CHECK_TRUE_RET(iter->second != nullptr, std::vector<opt::PassPtr>{});
    schedule_passes.push_back(iter->second);
  }
  return schedule_passes;
}
}  // namespace registry
}  // namespace mindspore
