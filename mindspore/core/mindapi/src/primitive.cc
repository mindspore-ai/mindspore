/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "mindapi/ir/primitive.h"
#include "mindapi/src/helper.h"
#include "ir/primitive.h"
#include "ir/value.h"

namespace mindspore::api {
using ValueImpl = mindspore::Value;
using PrimitiveImpl = mindspore::Primitive;

MIND_API_BASE_IMPL(Primitive, PrimitiveImpl, Value);

Primitive::Primitive(const std::string &name) : Value(std::make_shared<PrimitiveImpl>(name)) {}

const std::string &Primitive::name() const { return ToRef<PrimitiveImpl>(impl_).name(); }

Primitive &Primitive::AddAttr(const std::string &name, const ValuePtr &attr) {
  auto value = ToImpl<ValueImpl>(attr);
  ToRef<PrimitiveImpl>(impl_).set_attr(name, value);
  return *this;
}

Primitive &Primitive::SetAttrs(const std::unordered_map<std::string, ValuePtr> &attrs) {
  for (auto &attr : attrs) {
    auto value = ToImpl<ValueImpl>(attr.second);
    ToRef<PrimitiveImpl>(impl_).set_attr(attr.first, value);
  }
  return *this;
}

void Primitive::EraseAttr(const std::string &name) { ToRef<PrimitiveImpl>(impl_).EraseAttr(name); }

ValuePtr Primitive::GetAttr(const std::string &name) const {
  auto v = ToRef<PrimitiveImpl>(impl_).GetAttr(name);
  return ToWrapper<Value>(v);
}

bool Primitive::HasAttr(const std::string &name) const { return ToRef<PrimitiveImpl>(impl_).HasAttr(name); }

std::unordered_map<std::string, ValuePtr> Primitive::attrs() const {
  std::unordered_map<std::string, ValuePtr> attr_map;
  auto &impl_attrs = ToRef<PrimitiveImpl>(impl_).attrs();
  attr_map.reserve(impl_attrs.size());
  for (auto &attr : impl_attrs) {
    auto value = ToWrapper<Value>(attr.second);
    (void)attr_map.emplace(attr.first, value);
  }
  return attr_map;
}
}  // namespace mindspore::api
