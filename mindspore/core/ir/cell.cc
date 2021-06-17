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

#include "ir/cell.h"

#include <utility>
#include <map>
#include <algorithm>

#include "abstract/abstract_value.h"

namespace mindspore {
using mindspore::abstract::AbstractFunction;

abstract::AbstractBasePtr Cell::ToAbstract() { return nullptr; }

bool Cell::operator==(const Value &other) const {
  if (other.isa<Cell>()) {
    auto other_prim = static_cast<const Cell &>(other);
    return *this == other_prim;
  } else {
    return false;
  }
}

bool Cell::operator==(const Cell &other) const {
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
    MS_EXCEPTION_IF_NULL(iter->second);
    return *item.second == *iter->second;
  });
  return all;
}

std::string Cell::GetAttrString() const {
  std::ostringstream buffer;
  bool begin = true;
  buffer << "{" << std::endl;
  for (auto &attr : attrs_) {
    if (!begin) {
      buffer << ", " << std::endl;
    } else {
      begin = false;
    }
    buffer << attr.first << ":" << attr.second->ToString();
  }
  buffer << "}";
  return buffer.str();
}

std::string Cell::ToString() const {
  std::ostringstream buffer;
  buffer << "Cell " << name();
  return buffer.str();
}

void Cell::DelAttr(const std::string &name) { attrs_.erase(name); }
}  // namespace mindspore
