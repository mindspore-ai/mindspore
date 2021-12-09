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

#include "mapper/op_mapper_registry.h"
#include <string>

namespace mindspore {
namespace dpico {
OpMapperRegistry::OpMapperRegistry() = default;

OpMapperRegistry *OpMapperRegistry::GetInstance() {
  static OpMapperRegistry instance;
  return &instance;
}

OpMapperPtr OpMapperRegistry::GetOpMapper(const std::string &name) {
  auto it = op_mappers_.find(name);
  if (it != op_mappers_.end()) {
    return it->second;
  }
  return nullptr;
}
}  // namespace dpico
}  // namespace mindspore
