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

#ifndef MINDSPORE_LITE_SRC_OPS_MS_OPS_UTILS_H_
#define MINDSPORE_LITE_SRC_OPS_MS_OPS_UTILS_H_

#include <map>
#include <string>
#include "src/ops/ops_func_declare.h"

#ifdef PRIMITIVE_WRITEABLE
namespace mindspore {
namespace lite {
typedef schema::PrimitiveT *(*PrimitiveTCreator)(const AnfNodePtr &node);

class MSOpsRegistry {
 public:
  static MSOpsRegistry *GetInstance() {
    static MSOpsRegistry registry;
    return &registry;
  }
  void InsertPrimitiveTMap(std::string name, PrimitiveTCreator creator) { primitive_creators[name] = creator; }
  PrimitiveTCreator GetPrimitiveCreator(std::string name) {
    if (primitive_creators.find(name) != primitive_creators.end()) {
      return primitive_creators[name];
    } else {
      MS_LOG(ERROR) << "Unsupported primitive type in Create: " << name;
      return nullptr;
    }
  }

 protected:
  std::map<std::string, PrimitiveTCreator> primitive_creators;
};

class RegistryMSOps {
 public:
  RegistryMSOps(std::string name, PrimitiveTCreator creator) {
    MSOpsRegistry::GetInstance()->InsertPrimitiveTMap(name, creator);
  }
  ~RegistryMSOps() = default;
};

schema::PrimitiveT *GetPrimitiveT(const mindspore::AnfNodePtr &node);
}  // namespace lite
}  // namespace mindspore
#endif

#endif  // MINDSPORE_LITE_SRC_OPS_MS_OPS_UTILS_H_
