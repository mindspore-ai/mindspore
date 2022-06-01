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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_MAPPER_OP_MAPPER_REGISTRY_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_MAPPER_OP_MAPPER_REGISTRY_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "mapper/op_mapper.h"

namespace mindspore {
namespace dpico {
class OpMapperRegistry {
 public:
  OpMapperRegistry();

  virtual ~OpMapperRegistry() = default;

  static OpMapperRegistry *GetInstance();

  OpMapperPtr GetOpMapper(const std::string &name);

  std::unordered_map<std::string, OpMapperPtr> op_mappers_;
};

class OpMapperRegistrar {
 public:
  OpMapperRegistrar(const std::string &op_type_name, const OpMapperPtr &op_mapper) {
    OpMapperRegistry::GetInstance()->op_mappers_[op_type_name] = op_mapper;
  }
  ~OpMapperRegistrar() = default;
};

#define REG_MAPPER(primitive_type, mapper) \
  static OpMapperRegistrar g_##primitive_type##MapperReg(#primitive_type, std::make_shared<mapper>());
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_MAPPER_OP_MAPPER_REGISTRY_H_
