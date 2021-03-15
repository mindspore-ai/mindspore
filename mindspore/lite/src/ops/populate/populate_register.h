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

#ifndef LITE_MINDSPORE_LITE_C_OPS_OP_POPULATE_REGISTER_H
#define LITE_MINDSPORE_LITE_C_OPS_OP_POPULATE_REGISTER_H

#include <map>
#include "schema/model_generated.h"
#include "nnacl/op_base.h"
#include "src/common/common.h"
#include "src/common/prim_util.h"

namespace mindspore {
namespace lite {
typedef OpParameter *(*ParameterGen)(const void *prim);
class PopulateRegistry {
 public:
  static PopulateRegistry *GetInstance() {
    static PopulateRegistry registry;
    return &registry;
  }

  void InsertParameterMap(int type, ParameterGen creator, int version) {
    parameters_[GenPrimVersionKey(type, version)] = creator;
  }

  ParameterGen GetParameterCreator(int type, int version) {
    ParameterGen param_creator = nullptr;
    auto iter = parameters_.find(GenPrimVersionKey(type, version));
    if (iter == parameters_.end()) {
      MS_LOG(ERROR) << "Unsupported parameter type in Create : "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
      return nullptr;
    }
    param_creator = iter->second;
    return param_creator;
  }

 protected:
  // key:type * 1000 + schema_version
  std::map<int, ParameterGen> parameters_;
};

class Registry {
 public:
  Registry(int primitive_type, ParameterGen creator, int version) {
    PopulateRegistry::GetInstance()->InsertParameterMap(primitive_type, creator, version);
  }
  ~Registry() = default;
};

}  // namespace lite
}  // namespace mindspore
#endif
