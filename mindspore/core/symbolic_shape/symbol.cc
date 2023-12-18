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

#include "mindspore/core/symbolic_shape/symbol.h"
#include <utility>
#include "mindspore/core/symbolic_shape/utils.h"

namespace mindspore {
namespace symshape {
size_t Symbol::id() const {
  if (id_ == 0) {
    static size_t cur_id = 0;
    id_ = ++cur_id;
  }
  return id_;
}

void DynamicSymbol::UpdateImpl(const SymbolPtr &s) {
  if (auto d = s->cast_ptr<DynamicSymbol>(); d != nullptr) {
    symbol_ = d->symbol_;
  } else {
    symbol_ = s;
  }
}

void ScalarSymbol::UpdateImpl(const SymbolPtr &s) {
  if (is_const_) {
    MS_LOG(EXCEPTION) << "Const symbol '" << ToString() << "' cannot be updated, other: " << s->ToString();
    return;
  }
  if (s->tid() != tid()) {
    MS_LOG(EXCEPTION) << "Symbol " << s->ToString() << " is not a " << type_name();
    return;
  }
  if (s->HasData()) {
    SetValueByScalar(s.get());
    has_data_ = true;
  }
}

bool ScalarSymbol::operator==(const Symbol &s) const {
  if (this == &s) {
    return true;
  }
  if (!has_data_ || !s.HasData() || s.tid() != tid()) {
    return false;
  }
  return CheckEqualValue(&s);
}

std::string BoolSymbol::ToRawString() const { return has_data_ ? (value_ ? "true" : "false") : sid(); }

std::string FloatSymbol::ToRawString() const { return has_data_ ? std::to_string(value_) : sid(); }

std::string StrSymbol::ToRawString() const { return has_data_ ? value_ : sid(); }

std::string IntSymbol::ToRawString() const { return has_data_ ? std::to_string(value_) : sid(); }

std::shared_ptr<IntSymbol> IntSymbol::Make(int64_t val, const OpPtr &op) {
  auto s = std::make_shared<IntSymbol>(true, true, op);
  s->value_ = val;
  s->SetRange(val, val);
  return s;
}

std::string IntSymbol::ToString() const {
  std::ostringstream oss;
  oss << ScalarSymbol::ToString();
  if (has_data_) {
    return oss.str();
  }
  math_info_.Dump(oss);
  return oss.str();
}

void IntSymbol::UpdateImpl(const SymbolPtr &s) {
  if (is_const_) {
    MS_LOG(EXCEPTION) << "Const symbol '" << ToString() << "' cannot be updated, other: " << s->ToString();
  }
  auto other = s->as<IntSymbol>();
  if (other == nullptr) {
    MS_LOG(EXCEPTION) << "Symbol " << s->ToString() << " is not a IntSymbol.";
  }
  if (!other->has_data_) {
    MS_LOG(EXCEPTION) << "Symbol " << s->ToString() << " has no data.";
  }
  value_ = other->value_;
  has_data_ = true;
}

bool IntSymbol::operator==(const Symbol &s) const {
  if (this == &s) {
    return true;
  }
  auto other = s.as<IntSymbol>();
  if (other == nullptr) {
    return false;
  }
  if (has_data_ && other->has_data_) {
    return value_ == other->value_;
  }
  return math_info_.MathEquals(other->math_info_);
}

bool IntSymbol::operator<(const IntSymbol &s) const {
  if (this == &s) {
    return false;
  }
  if (has_data_ && s.has_data_) {
    return value_ < s.value_;
  }
  return math_info_.MathLess(s.math_info_);
}
bool IntSymbol::operator<=(const IntSymbol &s) const {
  if (this == &s) {
    return true;
  }
  if (has_data_ && s.has_data_) {
    return value_ <= s.value_;
  }
  return math_info_.MathLessEqual(s.math_info_);
}

bool ListSymbol::operator==(const Symbol &s) const {
  if (this == &s) {
    return true;
  }
  if (!has_data_ || !s.HasData()) {
    return false;
  }
  auto *list = s.as<ListSymbol>();
  if (size() != list->size()) {
    return false;
  }
  for (size_t i = 0; i < symbols_.size(); i++) {
    if (!symbols_[i]->EqualsTo(list->symbols_[i])) {
      return false;
    }
  }
  return true;
}

void ListSymbol::UpdateList(const SymbolPtrList &slist) {
  has_data_ = true;
  if (is_dyn_len_) {
    symbols_ = slist;
  } else {
    if (size() != slist.size()) {
      MS_LOG(EXCEPTION) << "Symbol " << ToString() << " size does not equals to the other symbol size. " << size()
                        << " vs " << slist.size();
    }
    for (size_t i = 0; i < symbols_.size(); i++) {
      if (symbols_[i]->CanUpdate()) {
        symbols_[i]->Update(slist[i]);
      }
    }
  }
}

std::string ListSymbol::ToString() const {
  if (!has_data_) {
    return "[DynLen-" + sid() + "]";
  }
  return SymbolListToStr(symbols_, "[", "]");
}

std::string ListSymbol::ToRawString() const { return SymbolListToStr(symbols_, "{", "}", true); }

void ListSymbol::UpdateImpl(const SymbolPtr &s) {
  ListSymbol *other = s->as<ListSymbol>();
  if (other == nullptr) {
    MS_LOG(EXCEPTION) << "Symbol " << s->ToString() << " is not a ListSymbol";
  }
  UpdateList(other->symbols());
}

const SymbolPtr &ListSymbol::item(size_t i) const {
  if (i >= symbols_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Index " << i << " out of range of symbols size " << symbols_.size();
  }
  return symbols_[i];
}
}  // namespace symshape
}  // namespace mindspore
