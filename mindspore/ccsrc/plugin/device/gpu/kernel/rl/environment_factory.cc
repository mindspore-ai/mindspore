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
#include "plugin/device/gpu/kernel/rl/environment_factory.h"
#include <utility>

namespace mindspore {
namespace kernel {
EnvironmentFactory &EnvironmentFactory::GetInstance() {
  static EnvironmentFactory instance;
  return instance;
}

std::tuple<int, std::shared_ptr<Environment>> EnvironmentFactory::Create(const std::string &name) {
  auto env_iter = map_env_name_to_creators_.find(name);
  if (env_iter == map_env_name_to_creators_.end()) {
    std::ostringstream oss;
    oss << "Environment " << name << " not exist.\n";
    oss << "Environment registered list: [";
    for (auto iter = map_env_name_to_creators_.begin(); iter != map_env_name_to_creators_.end(); iter++) {
      oss << iter->first << " ";
    }
    oss << "]";
    MS_LOG(EXCEPTION) << oss.str();
  }

  auto env = std::shared_ptr<Environment>(env_iter->second());
  map_env_handle_to_instances_.insert(std::make_pair(++handle_, env));
  return std::make_tuple(handle_, env);
}

void EnvironmentFactory::Delete(int64_t handle) { map_env_handle_to_instances_.erase(handle); }

std::shared_ptr<Environment> EnvironmentFactory::GetByHandle(int64_t handle) {
  auto iter = map_env_handle_to_instances_.find(handle);
  if (iter == map_env_handle_to_instances_.end()) {
    MS_LOG(EXCEPTION) << "Environment " << handle << " not exist.";
  }

  return iter->second;
}

void EnvironmentFactory::Register(const std::string &name, EnvCreator &&creator) {
  map_env_name_to_creators_.insert(std::make_pair(name, creator));
}
}  // namespace kernel
}  // namespace mindspore
