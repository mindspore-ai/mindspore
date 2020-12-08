/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ps/ps_cache/ps_cache_factory.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
PsCacheFactory &PsCacheFactory::Get() {
  static PsCacheFactory instance;
  return instance;
}

void PsCacheFactory::Register(const std::string &device_name, PsCacheCreator &&ps_cache_creator) {
  if (ps_cache_creators_.end() == ps_cache_creators_.find(device_name)) {
    (void)ps_cache_creators_.emplace(device_name, ps_cache_creator);
  }
}

std::shared_ptr<PsCacheBasic> PsCacheFactory::ps_cache(const std::string &device_name) {
  auto iter = ps_cache_creators_.find(device_name);
  if (ps_cache_creators_.end() != iter) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return (iter->second)();
  }
  return nullptr;
}
}  // namespace ps
}  // namespace mindspore
