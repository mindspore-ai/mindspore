/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "mindapi/ir/type.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ir/dtype.h"
#include "ir/dtype/type.h"
#include "ir/dtype/tensor_type.h"
#include "abstract/utils.h"

namespace mindspore::api {
using TypeImpl = mindspore::Type;
using TensorTypeImpl = mindspore::TensorType;

MIND_API_BASE_IMPL(Type, TypeImpl, Value);

TypeId Type::type_id() const { return ToRef<TypeImpl>(impl_).type_id(); }

TypeId Type::number_type() const { return ToRef<TypeImpl>(impl_).number_type(); }

TypePtr Type::GetType(TypeId id) {
  auto type_impl = mindspore::TypeIdToType(id);
  return ToWrapper<Type>(type_impl);
}

size_t Type::GetSize(TypeId id) { return mindspore::abstract::TypeIdSize(id); }

MIND_API_BASE_IMPL(TensorType, TensorTypeImpl, Type);

TensorType::TensorType(const TypePtr &element_type)
    : Type(std::make_shared<TensorTypeImpl>(ToImpl<TypeImpl>(element_type))) {}

TypePtr TensorType::element() const {
  auto element_type_impl = ToRef<TensorTypeImpl>(impl_).element();
  return ToWrapper<Type>(element_type_impl);
}
}  // namespace mindspore::api
