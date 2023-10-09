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

#include <utility>
#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"

namespace mindspore::graphkernel::symbol {
void ScalarSymbol::UpdateImpl(const SymbolPtr &s) {
  if (is_const_) {
    MS_LOG(EXCEPTION) << "Const symbol '" << ToString() << "' cannot be updated, other: " << s->ToString();
    return;
  }
  ScalarSymbol *other = s->as<ScalarSymbol>();
  if (other == nullptr || other->tid() != tid()) {
    MS_LOG(EXCEPTION) << "Symbol " << s->ToString() << " is not a " << type_name();
    return;
  }
  if (other->has_data_) {
    SetValueByScalar(other);
    has_data_ = true;
  }
}

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
  PrintMathInfo(oss);
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
  auto p = s.as<IntSymbol>();
  if (!has_data_ || !s.HasData()) {
    if (p != nullptr && !has_data_ && !s.HasData()) {
      return MathEquals(*p);
    }
    return false;
  }
  return p != nullptr && p->value_ == value_;
}

bool IntSymbol::operator<(const IntSymbol &s) const {
  if (this == &s) {
    return false;
  }
  if (has_data_ && s.has_data_) {
    return value_ < s.value_;
  }
  return MathLess(s);
}
bool IntSymbol::operator<=(const IntSymbol &s) const {
  if (this == &s) {
    return true;
  }
  if (has_data_ && s.has_data_) {
    return value_ <= s.value_;
  }
  return MathLessEqual(s);
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

void ListSymbol::UpdateImpl(const SymbolPtr &s) {
  ListSymbol *other = s->as<ListSymbol>();
  if (other == nullptr) {
    MS_LOG(EXCEPTION) << "Symbol " << s->ToString() << " is not a ListSymbol";
  }
  UpdateList(other->symbols());
}

IListSymbol::SPtr IListSymbol::FromShape(const ShapeVector &shape, bool real_value, const OpPtr &op) {
  if (!real_value && IsDynamicRank(shape)) {
    return IListSymbol::Make(op);
  }
  SymbolPtrList result(shape.size());
  (void)std::transform(shape.begin(), shape.end(), result.begin(), [real_value, op](int64_t s) {
    if (!real_value && s == abstract::Shape::kShapeDimAny) {
      return IntSymbol::Make(op);
    } else {
      return IntSymbol::Make(s, op);
    }
  });
  return IListSymbol::Make(std::move(result), op);
}

std::string SymbolListToStr(const SymbolPtrList &slist, const std::string &pre, const std::string &post, bool expr) {
  std::ostringstream oss;
  oss << pre;
  bool first = true;
  for (auto &s : slist) {
    if (first) {
      first = false;
    } else {
      oss << ", ";
    }
    oss << (expr ? s->ToExpr() : s->ToString());
  }
  oss << post;
  return oss.str();
}
}  // namespace mindspore::graphkernel::symbol
