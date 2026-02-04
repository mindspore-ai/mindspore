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

#include "abstract/symbolic_shape/symbol.h"
#include <utility>
#include "ir/dtype/tensor_type.h"
#include "abstract/symbolic_shape/utils.h"
#include "abstract/symbolic_shape/int_symbol.h"

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
ValuePtr BoolSymbol::ToValue() const { return has_data_ ? MakeValue<bool>(value_) : kValueAny; }

std::string FloatSymbol::ToRawString() const { return has_data_ ? std::to_string(value_) : sid(); }
ValuePtr FloatSymbol::ToValue() const { return has_data_ ? MakeValue<double>(value_) : kValueAny; }
ValuePtr FloatSymbol::ToValueOf(const TypePtr &type) const {
  if (!has_data_) {
    return kValueAny;
  }
  auto type_id = type->type_id();
  if (type_id == kObjectTypeTensorType) {
    auto tensor_type = type->cast_ptr<TensorType>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    type_id = tensor_type->element()->type_id();
  }
  switch (type_id) {
    case kNumberTypeFloat64:
      return MakeValue<double>(value_);
    case kNumberTypeFloat32:
      return MakeValue<float>(static_cast<float>(value_));
    default:
      MS_LOG(INTERNAL_EXCEPTION) << "Cannot convert the FloatSymbol to type " << type->ToString();
  }
  return ToValue();
}

std::string StrSymbol::ToRawString() const { return has_data_ ? value_ : sid(); }
ValuePtr StrSymbol::ToValue() const { return has_data_ ? MakeValue<std::string>(value_) : kValueAny; }
}  // namespace symshape
}  // namespace mindspore
