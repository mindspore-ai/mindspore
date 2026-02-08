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
#include "abstract/symbolic_shape/symbol.h"
#include "abstract/symbolic_shape/utils.h"
#include "abstract/symbolic_shape/int_symbol.h"
#include "ir/tensor_new.h"
#include "ir/dtype/tensor_type.h"
#include "mindspore/core/abstract/symbolic_shape/symbol_utils.h"

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

template <typename T, typename S>
ValuePtr ToTensor(TypeId type, const SymbolPtrList &symbols) {
  std::vector<T> values;
  values.reserve(symbols.size());
  (void)std::transform(symbols.begin(), symbols.end(), std::back_inserter(values),
                       [](const SymbolPtr &s) { return static_cast<T>(s->as<S>()->value()); });
  return tensor::from_buffer(type, ShapeVector{static_cast<int64_t>(values.size())}, values.data(), type);
}

ValuePtr ToTensorOf(const TypePtr &type, const SymbolPtrList &symbols) {
  auto tensor_type = type->cast_ptr<TensorType>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto type_id = tensor_type->element()->type_id();
  switch (type_id) {
    // IntSymbol
    case kNumberTypeInt64:
      return ToTensor<int64_t, IntSymbol>(type_id, symbols);
    case kNumberTypeInt32:
      return ToTensor<int32_t, IntSymbol>(type_id, symbols);
    case kNumberTypeInt16:
      return ToTensor<int16_t, IntSymbol>(type_id, symbols);
    case kNumberTypeInt8:
      return ToTensor<int8_t, IntSymbol>(type_id, symbols);
    case kNumberTypeUInt64:
      return ToTensor<uint64_t, IntSymbol>(type_id, symbols);
    case kNumberTypeUInt32:
      return ToTensor<uint32_t, IntSymbol>(type_id, symbols);
    case kNumberTypeUInt16:
      return ToTensor<uint16_t, IntSymbol>(type_id, symbols);
    case kNumberTypeUInt8:
      return ToTensor<uint8_t, IntSymbol>(type_id, symbols);
    // FloatSymbol
    case kNumberTypeFloat64:
      return ToTensor<double, FloatSymbol>(type_id, symbols);
    case kNumberTypeFloat32:
      return ToTensor<float, FloatSymbol>(type_id, symbols);
    default:
      break;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Cannot convert the symbols to tensor. type: " << type->ToString();
}

ValuePtr ListSymbol::ToValueOf(const TypePtr &type) const {
  if (!AllHaveData()) {
    return kValueAny;
  }
  if (type->isa<TensorType>()) {
    return ToTensorOf(type, symbols());
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
