/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "abstract/abstract_value.h"
#include "utils/ms_utils.h"

namespace mindspore {
static std::string MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return "C" + std::to_string(last_id.fetch_add(1, std::memory_order_relaxed));
}

using mindspore::abstract::AbstractFunction;

abstract::AbstractBasePtr Cell::ToAbstract() { return nullptr; }

Cell::Cell(const std::string &name) : Named(name), id_(MakeId()) {}

bool Cell::operator==(const Value &other) const {
  if (other.isa<Cell>()) {
    auto &other_prim = static_cast<const Cell &>(other);
    return *this == other_prim;
  } else {
    return false;
  }
}

bool Cell::operator==(const Cell &other) const {
  if (name() != other.name()) {
    return false;
  }
  return common::IsAttrsEqual(attrs_, other.attrs_);
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
