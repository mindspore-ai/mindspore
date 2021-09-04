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

#ifndef ACL_MAPPER_PRIMITIVE_MAPPER_REGISTER_H
#define ACL_MAPPER_PRIMITIVE_MAPPER_REGISTER_H

#include <map>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "tools/converter/acl/mapper/primitive_mapper.h"

namespace mindspore {
namespace lite {
class PrimitiveMapperRegister {
 public:
  static PrimitiveMapperRegister &GetInstance();

  void InsertPrimitiveMapper(const std::string &name, const PrimitiveMapperPtr &deparser);

  PrimitiveMapperPtr GetPrimitiveMapper(const std::string &name);

 private:
  PrimitiveMapperRegister() = default;
  ~PrimitiveMapperRegister() = default;

  std::map<std::string, PrimitiveMapperPtr> deparser_;
};

class RegisterPrimitiveMapper {
 public:
  RegisterPrimitiveMapper(const std::string &name, const PrimitiveMapperPtr &deparser);

  ~RegisterPrimitiveMapper() = default;
};

#define REGISTER_PRIMITIVE_MAPPER(name, mapper) \
  static RegisterPrimitiveMapper g_##name##PrimMapper(name, std::make_shared<mapper>());
}  // namespace lite
}  // namespace mindspore
#endif  // ACL_MAPPER_PRIMITIVE_MAPPER_REGISTER_H
