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

#ifndef MINDSPORE_LITE_SRC_COMMON_OPS_OPS_UTILS_H_
#define MINDSPORE_LITE_SRC_COMMON_OPS_OPS_UTILS_H_

#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include "src/common/ops/ops_func_declare.h"
#ifdef PRIMITIVE_WRITEABLE
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
typedef std::unique_ptr<schema::PrimitiveT> (*PrimitiveTCreator)(const PrimitivePtr &primitive);

class MSOpsRegistry {
 public:
  static MSOpsRegistry *GetInstance() {
    static MSOpsRegistry registry;
    return &registry;
  }
  void InsertPrimitiveTMap(const std::string &name, PrimitiveTCreator creator) {
    std::string lower_name = name;
    (void)std::transform(name.begin(), name.end(), lower_name.begin(), ::tolower);
    primitive_creators[lower_name] = creator;
  }
  PrimitiveTCreator GetPrimitiveCreator(const std::string &name) {
    std::string lower_name = name;
    (void)std::transform(name.begin(), name.end(), lower_name.begin(), ::tolower);
    (void)lower_name.erase(std::remove(lower_name.begin(), lower_name.end(), '_'), lower_name.end());
    if (primitive_creators.find(lower_name) != primitive_creators.end()) {
      return primitive_creators[lower_name];
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
  RegistryMSOps(const std::string &name, PrimitiveTCreator creator) noexcept {
    MSOpsRegistry::GetInstance()->InsertPrimitiveTMap(name, creator);
  }
  ~RegistryMSOps() = default;
};

#define REG_MINDSPORE_OPERATOR(OP) \
  static RegistryMSOps g_##OP##PrimitiveCreatorRegistry(#OP, PrimitiveCreator<mindspore::ops::OP>);
}  // namespace lite
}  // namespace mindspore
#endif

#endif  // MINDSPORE_LITE_SRC_COMMON_OPS_OPS_UTILS_H_
