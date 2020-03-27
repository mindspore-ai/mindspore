/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ir/base.h"
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace mindspore {
const bool Base::IsFromTypeId(uint32_t tid) const {
  static const uint32_t node_id = GetTypeId(typeid(Base).name());
  return tid == node_id;
}

uint32_t Base::GetTypeId(const char *const type_name) {
  TypeIdManager *t = TypeIdManager::Get();
  std::lock_guard<std::mutex>(t->mutex);
  auto it = t->map.find(type_name);
  if (it != t->map.end()) {
    return it->second;
  }
  uint32_t tid = ++(t->type_counter);
  t->map[type_name] = tid;
  return tid;
}
}  // namespace mindspore
