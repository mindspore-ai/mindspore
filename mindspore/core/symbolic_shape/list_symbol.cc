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
#include "mindspore/core/symbolic_shape/symbol.h"
#include "mindspore/core/symbolic_shape/utils.h"
#include "mindspore/core/symbolic_shape/int_symbol.h"

namespace mindspore {
namespace symshape {
bool ListSymbol::operator==(const Symbol &s) const {
  if (this == &s) {
    return true;
  }
  if (!has_data_ || !s.HasData()) {
    return false;
  }
  auto *list = s.as_noexcept<ListSymbol>();
  if (list == nullptr || size() != list->size()) {
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

ValuePtr ListSymbol::ToValue() const {
  if (!AllHaveData()) {
    return kValueAny;
  }
  ValuePtrList values;
  values.reserve(symbols_.size());
  (void)std::transform(symbols_.begin(), symbols_.end(), std::back_inserter(values),
                       [](const SymbolPtr &s) { return s->ToValue(); });
  return std::make_shared<ValueTuple>(values);
}

ValuePtr ListSymbol::ToValueOf(const TypePtr &type) const {
  if (!AllHaveData()) {
    return kValueAny;
  }
  ValuePtrList values;
  values.reserve(symbols_.size());
  TypePtr inner_type = type;
  if (type->isa<Tuple>()) {
    auto tuple = type->cast_ptr<Tuple>();
    if (tuple->dynamic_len()) {
      inner_type = tuple->dynamic_element_type();
    } else if (!tuple->elements().empty()) {
      // element type in tuple is all the same.
      inner_type = tuple->elements()[0];
    }
  }
  (void)std::transform(symbols_.begin(), symbols_.end(), std::back_inserter(values),
                       [&inner_type](const SymbolPtr &s) { return s->ToValueOf(inner_type); });
  return std::make_shared<ValueTuple>(values);
}

void ListSymbol::UpdateImpl(const SymbolPtr &s) { UpdateList(s->as<ListSymbol>()->symbols()); }

const SymbolPtr &ListSymbol::item(size_t i) const {
  if (MS_UNLIKELY(i >= symbols_.size())) {
    MS_LOG(INTERNAL_EXCEPTION) << "Index " << i << " out of range of symbols size " << symbols_.size();
  }
  return symbols_[i];
}
}  // namespace symshape
}  // namespace mindspore
