/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "utils/symbolic.h"

#include <memory>

namespace mindspore {
std::ostream &operator<<(std::ostream &out, const std::shared_ptr<EnvInstance> &objPtr) {
  out << "(";
  MS_EXCEPTION_IF_NULL(objPtr);
  for (auto &iter : objPtr->contents_) {
    out << iter.first << ":" << iter.second << ";";
  }
  out << ")";
  return out;
}

bool EnvInstance::operator==(const EnvInstance &other) const {
  if (Len() != other.Len()) {
    return false;
  }
  bool equal = std::all_of(contents_.begin(), contents_.end(),
                           [&other](const std::pair<SymbolicKeyInstancePtr, Any> &item) -> bool {
                             return other.contents_.find(item.first) != other.contents_.end();
                           });
  return equal;
}
bool EnvInstance::operator==(const Value &other) const {
  if (other.isa<EnvInstance>()) {
    auto other_env_inst = static_cast<const EnvInstance *>(&other);
    return *this == *other_env_inst;
  }
  return false;
}
std::shared_ptr<EnvInstance> newenv = std::make_shared<EnvInstance>();
}  // namespace mindspore
