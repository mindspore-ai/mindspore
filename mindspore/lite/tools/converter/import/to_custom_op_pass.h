/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_TO_CUSTOM_OP_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_TO_CUSTOM_OP_PASS_H_
#include <string>
#include <map>
#include "include/backend/optimizer/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {

typedef int (*ToCustomOpFunc)(const CNodePtr &cnode);
class ToCustomOpRegistry {
 public:
  static ToCustomOpRegistry *GetInstance() {
    static ToCustomOpRegistry registry;
    return &registry;
  }

  void InsertToCustomOpMap(const std::string &key, ToCustomOpFunc creator) { to_custom_op_funcs_[key] = creator; }

  ToCustomOpFunc GetToCustomOpFunc(const std::string &key) {
    if (to_custom_op_funcs_.find(key) != to_custom_op_funcs_.end()) {
      return to_custom_op_funcs_[key];
    } else {
      MS_LOG(DEBUG) << "Unsupported primitive type : " << key;
      return nullptr;
    }
  }

 protected:
  std::map<std::string, ToCustomOpFunc> to_custom_op_funcs_;
};

class RegistryToCustomOp {
 public:
  RegistryToCustomOp(const std::string &key, ToCustomOpFunc creator) {
    ToCustomOpRegistry::GetInstance()->InsertToCustomOpMap(key, creator);
  }
  virtual ~RegistryToCustomOp() = default;
};

#define REGISTER_TO_CUSTOM_OP(type, to_custom_op_func) \
  RegistryToCustomOp g_##type##_to_custom_op(type, to_custom_op_func);

class ToCustomOpPass : public Pass {
 public:
  ToCustomOpPass() : Pass("ToCustomOpPass") {}
  ~ToCustomOpPass() = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_TO_CUSTOM_OP_PASS_H_
