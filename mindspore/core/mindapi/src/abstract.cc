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

#include "mindapi/ir/abstract.h"
#include "mindapi/src/helper.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "ir/dtype.h"

namespace mindspore::api {
using TypeImpl = mindspore::Type;
using ValueImpl = mindspore::Value;
using ShapeImpl = mindspore::abstract::Shape;
using AbstractBaseImpl = mindspore::abstract::AbstractBase;

MIND_API_BASE_IMPL(AbstractBase, AbstractBaseImpl, Base);

AbstractBasePtr AbstractBase::Clone() const {
  auto abs = ToRef<AbstractBaseImpl>(impl_).Clone();
  return ToWrapper<AbstractBase>(abs);
}

TypePtr AbstractBase::type() const {
  auto t = ToRef<AbstractBaseImpl>(impl_).BuildType();
  return ToWrapper<Type>(t);
}

ValuePtr AbstractBase::value() const {
  auto v = ToRef<AbstractBaseImpl>(impl_).BuildValue();
  return ToWrapper<Value>(v);
}

ShapePtr AbstractBase::shape() const {
  auto s = ToRef<AbstractBaseImpl>(impl_).GetShapeTrack();
  return ToWrapper<Shape>(s);
}

void AbstractBase::set_type(const TypePtr &type) {
  auto type_impl = ToImpl<TypeImpl>(type);
  ToRef<AbstractBaseImpl>(impl_).set_type(type_impl);
}

void AbstractBase::set_value(const ValuePtr &value) {
  auto value_impl = ToImpl<ValueImpl>(value);
  ToRef<AbstractBaseImpl>(impl_).set_value(value_impl);
}

void AbstractBase::set_shape(const ShapePtr &shape) {
  auto shape_impl = ToImpl<ShapeImpl>(shape);
  ToRef<AbstractBaseImpl>(impl_).set_shape(shape_impl);
}

using AbstractScalarImpl = mindspore::abstract::AbstractScalar;

MIND_API_BASE_IMPL(AbstractScalar, AbstractScalarImpl, AbstractBase);

AbstractScalar::AbstractScalar(const ValuePtr &value, const TypePtr &type)
    : AbstractBase(std::make_shared<AbstractScalarImpl>(ToImpl<ValueImpl>(value), ToImpl<TypeImpl>(type))) {}

AbstractScalar::AbstractScalar(const TypePtr &type)
    : AbstractBase(std::make_shared<AbstractScalarImpl>(ToImpl<TypeImpl>(type))) {}

AbstractScalar::AbstractScalar(const ValuePtr &value)
    : AbstractBase(std::make_shared<AbstractScalarImpl>(ToImpl<ValueImpl>(value))) {}

AbstractScalar::AbstractScalar(int64_t value) : AbstractBase(std::make_shared<AbstractScalarImpl>(value)) {}

AbstractScalar::AbstractScalar(float value) : AbstractBase(std::make_shared<AbstractScalarImpl>(value)) {}

AbstractScalar::AbstractScalar(bool value) : AbstractBase(std::make_shared<AbstractScalarImpl>(value)) {}

AbstractScalar::AbstractScalar(const std::string &value) : AbstractBase(std::make_shared<AbstractScalarImpl>(value)) {}

using AbstractTensorImpl = mindspore::abstract::AbstractTensor;

MIND_API_BASE_IMPL(AbstractTensor, AbstractTensorImpl, AbstractBase);

AbstractTensor::AbstractTensor(TypeId type, const ShapeVector &shape)
    : AbstractBase(std::make_shared<AbstractTensorImpl>(mindspore::TypeIdToType(type), shape)) {}

AbstractBasePtr AbstractTensor::element() const {
  auto abs = ToRef<AbstractTensorImpl>(impl_).element();
  return ToWrapper<AbstractBase>(abs);
}

using AbstractSequenceImpl = mindspore::abstract::AbstractSequence;

MIND_API_BASE_IMPL(AbstractSequence, AbstractSequenceImpl, AbstractBase);

AbstractBasePtrList AbstractSequence::elements() const {
  auto &impl_elements = ToRef<AbstractSequenceImpl>(impl_).elements();
  return ToWrapperVector<AbstractBase>(impl_elements);
}

using AbstractTupleImpl = mindspore::abstract::AbstractTuple;

MIND_API_BASE_IMPL(AbstractTuple, AbstractTupleImpl, AbstractSequence);

AbstractTuple::AbstractTuple(const AbstractBasePtrList &elements)
    : AbstractSequence(std::make_shared<AbstractTupleImpl>(ToImplVector<AbstractBaseImpl>(elements))) {}
}  // namespace mindspore::api
