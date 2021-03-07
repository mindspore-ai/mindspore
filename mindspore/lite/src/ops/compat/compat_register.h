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

#ifndef LITE_MINDSPORE_LITE_C_OPS_OP_COMPAT_REGISTER_H_
#define LITE_MINDSPORE_LITE_C_OPS_OP_COMPAT_REGISTER_H_

#include <unordered_map>
#include <string>
#include <vector>
#include "include/model.h"
#include "schema/model_generated.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {
// compatibility, transfer attr to input tensor.
typedef int (*TransferAttrFunc)(Model::Node *node, std::vector<schema::Tensor *> *tensor,
                                std::vector<char *> *tensor_bufs);
class CompatRegistry {
 public:
  static CompatRegistry *GetInstance() {
    static CompatRegistry registry;
    return &registry;
  }

  void InsertTransferAttrFuncMap(int schema_version, int primitive_type, TransferAttrFunc transfer_attr_func) {
    int key = primitive_type * 10 + schema_version;
    transfer_attr_funcs_[key] = transfer_attr_func;
  }

  TransferAttrFunc GetTransferAttrFunc(int schema_version, int primitive_type) {
    int key = primitive_type * 10 + schema_version;
    if (transfer_attr_funcs_.find(key) != transfer_attr_funcs_.end()) {
      return transfer_attr_funcs_[key];
    } else {
      MS_LOG(DEBUG) << "Unsupported transformer type in Create : " << key;
      return nullptr;
    }
  }

 protected:
  std::unordered_map<int, TransferAttrFunc> transfer_attr_funcs_;
};

class Register {
 public:
  Register(int schema_version, int primitive_type, TransferAttrFunc transfer_attr_func) {
    CompatRegistry::GetInstance()->InsertTransferAttrFuncMap(schema_version, primitive_type, transfer_attr_func);
  }
  virtual ~Register() = default;
};
}  // namespace lite
}  // namespace mindspore
#endif  // LITE_MINDSPORE_LITE_C_OPS_OP_COMPAT_REGISTER_H_
