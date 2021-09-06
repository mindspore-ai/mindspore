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

#include "tools/converter/acl/mapper/primitive_mapper_register.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {
PrimitiveMapperRegister &PrimitiveMapperRegister::GetInstance() {
  static PrimitiveMapperRegister instance;
  return instance;
}

void PrimitiveMapperRegister::InsertPrimitiveMapper(const std::string &name, const PrimitiveMapperPtr &deparser) {
  deparser_[name] = deparser;
}

PrimitiveMapperPtr PrimitiveMapperRegister::GetPrimitiveMapper(const std::string &name) {
  if (deparser_.find(name) != deparser_.end()) {
    return deparser_[name];
  } else {
    MS_LOG(DEBUG) << "Unsupported primitive name : " << name;
    return nullptr;
  }
}

RegisterPrimitiveMapper::RegisterPrimitiveMapper(const std::string &name, const PrimitiveMapperPtr &deparser) {
  PrimitiveMapperRegister::GetInstance().InsertPrimitiveMapper(name, deparser);
}
}  // namespace lite
}  // namespace mindspore
