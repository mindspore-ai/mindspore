/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ir/primitive_base.h"

#include <utility>

namespace mindspore {
bool Primitive::operator==(const Value &other) const {
  if (other.isa<Primitive>()) {
    auto other_prim = static_cast<const Primitive &>(other);
    return *this == other_prim;
  } else {
    return false;
  }
}

bool Primitive::operator==(const Primitive &other) const {
  if (name() != other.name()) {
    return false;
  }
  if (attrs_.size() != other.attrs_.size()) {
    return false;
  }
  auto all = std::all_of(attrs_.begin(), attrs_.end(), [&other](const std::pair<std::string, ValuePtr> &item) -> bool {
    if (item.second == nullptr) {
      return false;
    }
    auto iter = other.attrs_.find(item.first);
    if (iter == other.attrs_.end()) {
      return false;
    }
    return *item.second == *iter->second;
  });
  return all;
}

std::string Primitive::GetAttrsText() const {
  if (attrs_.empty()) {
    return "";
  }

  std::ostringstream oss;
  oss << "[";
  bool is_first = true;
  for (auto &attr : attrs_) {
    if (is_first) {
      is_first = false;
    } else {
      oss << ", ";
    }
    oss << attr.first << "=" << attr.second->DumpText();
  }
  oss << "]";

  return oss.str();
}
}  // namespace mindspore
