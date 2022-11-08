/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
constexpr size_t kPassNumLimit = 10000;
std::map<std::string, PassBasePtr> outer_pass_storage;
std::map<registry::PassPosition, std::vector<std::string>> external_assigned_passes;
std::mutex pass_mutex;
void RegPass(const std::string &pass_name, const PassBasePtr &pass) {
  if (pass == nullptr) {
    MS_LOG(ERROR) << "pass is nullptr.";
    return;
  }
  std::unique_lock<std::mutex> lock(pass_mutex);
  if (outer_pass_storage.size() == kPassNumLimit) {
    MS_LOG(WARNING) << "pass's number is up to the limitation. The pass will not be registered.";
    return;
  }
  outer_pass_storage[pass_name] = pass;
}
}  // namespace

PassRegistry::PassRegistry(const std::vector<char> &pass_name, const PassBasePtr &pass) {
  RegPass(CharToString(pass_name), pass);
}

PassRegistry::PassRegistry(PassPosition position, const std::vector<std::vector<char>> &names) {
  if (position < POSITION_BEGIN || position > POSITION_END) {
    MS_LOG(ERROR) << "ILLEGAL position: position must be POSITION_BEGIN or POSITION_END.";
    return;
  }
  std::unique_lock<std::mutex> lock(pass_mutex);
  external_assigned_passes[position] = VectorCharToString(names);
}

std::vector<std::vector<char>> PassRegistry::GetOuterScheduleTaskInner(PassPosition position) {
  MS_CHECK_TRUE_MSG(position == POSITION_END || position == POSITION_BEGIN, {},
                    "position must be POSITION_END or POSITION_BEGIN.");
  return VectorStringToChar(external_assigned_passes[position]);
}

PassBasePtr PassRegistry::GetPassFromStoreRoom(const std::vector<char> &pass_name_char) {
  std::string pass_name = CharToString(pass_name_char);
  return outer_pass_storage.find(pass_name) == outer_pass_storage.end() ? nullptr : outer_pass_storage[pass_name];
}
}  // namespace registry
}  // namespace mindspore
