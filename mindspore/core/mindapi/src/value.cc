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

#include "mindapi/ir/value.h"
#include "mindapi/ir/type.h"
#include "mindapi/ir/abstract.h"
#include "mindapi/src/helper.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "ir/scalar.h"

namespace mindspore::api {
using ValueImpl = mindspore::Value;
using ValueSequenceImpl = mindspore::ValueSequence;
using ValueTupleImpl = mindspore::ValueTuple;
using StringImmImpl = mindspore::StringImm;
using ScalarImpl = mindspore::Scalar;
using BoolImmImpl = mindspore::BoolImm;
using IntegerImmImpl = mindspore::IntegerImm;
using Int8ImmImpl = mindspore::Int8Imm;
using Int16ImmImpl = mindspore::Int16Imm;
using Int32ImmImpl = mindspore::Int32Imm;
using Int64ImmImpl = mindspore::Int64Imm;
using UInt8ImmImpl = mindspore::UInt8Imm;
using FloatImmImpl = mindspore::FloatImm;
using FP32ImmImpl = mindspore::FP32Imm;

MIND_API_BASE_IMPL(Value, ValueImpl, Base);

TypePtr Value::type() const {
  auto t = ToRef<ValueImpl>(impl_).type();
  return ToWrapper<Type>(t);
}

AbstractBasePtr Value::ToAbstract() const {
  auto abs = ToRef<ValueImpl>(impl_).ToAbstract();
  return ToWrapper<AbstractBase>(abs);
}

MIND_API_BASE_IMPL(ValueSequence, ValueSequenceImpl, Value);

std::size_t ValueSequence::size() const { return ToRef<ValueSequenceImpl>(impl_).size(); }

std::vector<ValuePtr> ValueSequence::value() const {
  auto &elements = ToRef<ValueSequenceImpl>(impl_).value();
  return ToWrapperVector<Value>(elements);
}

MIND_API_BASE_IMPL(ValueTuple, ValueTupleImpl, ValueSequence);

ValueTuple::ValueTuple(const std::vector<ValuePtr> &elements)
    : ValueSequence(std::make_shared<ValueTupleImpl>(ToImplVector<ValueImpl>(elements))) {}

MIND_API_BASE_IMPL(StringImm, StringImmImpl, Value);

StringImm::StringImm(const std::string &str) : Value(std::make_shared<StringImmImpl>(str)) {}

const std::string &StringImm::value() const { return ToRef<StringImmImpl>(impl_).value(); }

MIND_API_BASE_IMPL(Scalar, ScalarImpl, Value);

MIND_API_BASE_IMPL(BoolImm, BoolImmImpl, Scalar);

BoolImm::BoolImm(bool b) : Scalar(std::make_shared<BoolImmImpl>(b)) {}

bool BoolImm::value() const { return ToRef<BoolImmImpl>(impl_).value(); }

MIND_API_BASE_IMPL(IntegerImm, IntegerImmImpl, Scalar);

MIND_API_BASE_IMPL(Int8Imm, Int8ImmImpl, IntegerImm);

Int8Imm::Int8Imm(int8_t value) : IntegerImm(std::make_shared<Int8ImmImpl>(value)) {}

int8_t Int8Imm::value() const { return ToRef<Int8ImmImpl>(impl_).value(); }

MIND_API_BASE_IMPL(Int16Imm, Int16ImmImpl, IntegerImm);

Int16Imm::Int16Imm(int16_t value) : IntegerImm(std::make_shared<Int16ImmImpl>(value)) {}

int16_t Int16Imm::value() const { return ToRef<Int16ImmImpl>(impl_).value(); }

MIND_API_BASE_IMPL(Int32Imm, Int32ImmImpl, IntegerImm);

Int32Imm::Int32Imm(int32_t value) : IntegerImm(std::make_shared<Int32ImmImpl>(value)) {}

int32_t Int32Imm::value() const { return ToRef<Int32ImmImpl>(impl_).value(); }

MIND_API_BASE_IMPL(Int64Imm, Int64ImmImpl, IntegerImm);

Int64Imm::Int64Imm(int64_t value) : IntegerImm(std::make_shared<Int64ImmImpl>(value)) {}

int64_t Int64Imm::value() const { return ToRef<Int64ImmImpl>(impl_).value(); }

MIND_API_BASE_IMPL(UInt8Imm, UInt8ImmImpl, IntegerImm);

UInt8Imm::UInt8Imm(uint8_t value) : UInt8Imm(std::make_shared<UInt8ImmImpl>(value)) {}

uint8_t UInt8Imm::value() const { return ToRef<UInt8ImmImpl>(impl_).value(); }

MIND_API_BASE_IMPL(FloatImm, FloatImmImpl, Scalar);

MIND_API_BASE_IMPL(FP32Imm, FP32ImmImpl, FloatImm);

FP32Imm::FP32Imm(float value) : FloatImm(std::make_shared<FP32ImmImpl>(value)) {}

float FP32Imm::value() const { return ToRef<FP32ImmImpl>(impl_).value(); }
}  // namespace mindspore::api
