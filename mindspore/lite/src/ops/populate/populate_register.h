/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef LITE_MINDSPORE_LITE_C_OPS_OP_POPULATE_REGISTER_H
#define LITE_MINDSPORE_LITE_C_OPS_OP_POPULATE_REGISTER_H

#include <map>
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {

class PopulateRegistry {
 public:
  static PopulateRegistry *GetInstance() {
    static PopulateRegistry registry;
    return &registry;
  }

  void InsertParameterMap(schema::PrimitiveType type, ParameterCreator creator) { parameter_creators[type] = creator; }

  ParameterCreator GetParameterCreator(schema::PrimitiveType type) {
    if (parameter_creators.find(type) != parameter_creators.end()) {
      return parameter_creators[type];
    } else {
      MS_LOG(ERROR) << "Unsupported parameter type in Create : " << schema::EnumNamePrimitiveType(type);
      return nullptr;
    }
  }

 protected:
  std::map<schema::PrimitiveType, ParameterCreator> parameter_creators;
};

class Registry {
 public:
  Registry(schema::PrimitiveType primitive_type, ParameterCreator creator) {
    PopulateRegistry::GetInstance()->InsertParameterMap(primitive_type, creator);
  }
  ~Registry() = default;
};

}  // namespace lite
}  // namespace mindspore
#endif
