/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/backend/debug/data_dump/overflow_dumper.h"

#ifndef ENABLE_SECURITY
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#endif

namespace mindspore {
namespace debug {
std::map<std::string, std::shared_ptr<OverflowDumper>> &OverflowDumper::GetInstanceMap() {
  static std::map<std::string, std::shared_ptr<OverflowDumper>> instance_map = {};
  return instance_map;
}

void OverflowDumper::Clear() { GetInstanceMap().clear(); }

std::shared_ptr<OverflowDumper> OverflowDumper::GetInstance(const std::string &name) noexcept {
  if (auto iter = GetInstanceMap().find(name); iter != GetInstanceMap().end()) {
    return iter->second;
  }
  return nullptr;
}

bool OverflowDumper::Register(const std::string &name, const std::shared_ptr<OverflowDumper> &instance) {
  if (GetInstanceMap().find(name) != GetInstanceMap().end()) {
    MS_LOG(WARNING) << name << " has been registered.";
  } else {
    (void)GetInstanceMap().emplace(name, instance);
  }
  return true;
}
}  // namespace debug
}  // namespace mindspore
