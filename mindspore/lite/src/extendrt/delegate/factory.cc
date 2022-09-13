/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/delegate/factory.h"

namespace mindspore {
DelegateRegistry &DelegateRegistry::GetInstance() {
  static DelegateRegistry instance;
  return instance;
}

void DelegateRegistry::RegDelegate(const mindspore::DeviceType &device_type, const std::string &provider,
                                   DelegateCreator creator) {
  auto it = creator_map_.find(device_type);
  if (it == creator_map_.end()) {
    HashMap<std::string, DelegateCreator> map;
    map[provider] = creator;
    creator_map_[device_type] = map;
    return;
  }
  it->second[provider] = creator;
}

void DelegateRegistry::UnRegDelegate(const mindspore::DeviceType &device_type, const std::string &provider) {
  auto it = creator_map_.find(device_type);
  if (it != creator_map_.end()) {
    creator_map_.erase(it);
  }
}

std::shared_ptr<GraphExecutor> DelegateRegistry::GetDelegate(const mindspore::DeviceType &device_type,
                                                             const std::string &provider,
                                                             const std::shared_ptr<Context> &ctx,
                                                             const ConfigInfos &config_infos) {
  //  find common delegate
  auto it = creator_map_.find(device_type);
  if (it == creator_map_.end()) {
    return nullptr;
  }
  auto creator_it = it->second.find(provider);
  if (creator_it == it->second.end()) {
    return nullptr;
  }
  return creator_it->second(ctx, config_infos);
}
}  // namespace mindspore
