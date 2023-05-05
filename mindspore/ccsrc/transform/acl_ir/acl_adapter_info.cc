/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/acl_ir/acl_adapter_info.h"

namespace mindspore {
namespace transform {
AclAdapterManager &AclAdapterManager::GetInstance() {
  static AclAdapterManager instance;
  return instance;
}

AclAdapterInfo &AclAdapterManager::Register(const std::string &op_type) {
  if (op_cache_.count(op_type) != 0) {
    return op_cache_.at(op_type);
  }

  op_cache_.emplace(op_type, AclAdapterInfo(op_type));
  return op_cache_.at(op_type);
}

bool AclAdapterManager::CheckAclAdapter(const std::string &op_type) {
  if (op_cache_.count(op_type) != 0) {
    return true;
  }
  return false;
}

const AclAdapterInfo &AclAdapterManager::GetOpInfo(const std::string &op_type) const {
  if (op_cache_.count(op_type) == 0) {
    MS_LOG(EXCEPTION) << "Current node " << op_type << " hasn't acl adapter";
  }
  return op_cache_.at(op_type);
}
}  // namespace transform
}  // namespace mindspore
