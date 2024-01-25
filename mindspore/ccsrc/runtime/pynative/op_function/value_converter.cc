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

#include "runtime/pynative/op_function/value_converter.h"

namespace mindspore::runtime {
Int64ImmPtr ValueConverter::ToInt(const ValuePtrList &inputs, size_t i) { return Convert<Int64ImmPtr>(inputs, i); }

FP32ImmPtr ValueConverter::ToFloat(const ValuePtrList &inputs, size_t i) { return Convert<FP32ImmPtr>(inputs, i); }

BoolImmPtr ValueConverter::ToBool(const ValuePtrList &inputs, size_t i) { return Convert<BoolImmPtr>(inputs, i); }

ScalarPtr ValueConverter::ToScalar(const ValuePtrList &inputs, size_t i) { return Convert<ScalarPtr>(inputs, i); }

tensor::TensorPtr ValueConverter::ToTensor(const ValuePtrList &inputs, size_t i) {
  return Convert<tensor::TensorPtr>(inputs, i);
}

StringImmPtr ValueConverter::ToString(const ValuePtrList &inputs, size_t i) { return Convert<StringImmPtr>(inputs, i); }

TypePtr ValueConverter::ToDtype(const ValuePtrList &inputs, size_t i) { return Convert<TypePtr>(inputs, i); }

ValueTuplePtr ValueConverter::ToValueTuple(const ValuePtrList &inputs, size_t i) {
  return Convert<ValueTuplePtr>(inputs, i);
}

std::optional<Int64ImmPtr> ValueConverter::ToIntOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<Int64ImmPtr>(inputs, i);
}

std::optional<FP32ImmPtr> ValueConverter::ToFloatOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<FP32ImmPtr>(inputs, i);
}

std::optional<BoolImmPtr> ValueConverter::ToBoolOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<BoolImmPtr>(inputs, i);
}

std::optional<ScalarPtr> ValueConverter::ToScalarOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<ScalarPtr>(inputs, i);
}

std::optional<tensor::TensorPtr> ValueConverter::ToTensorOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<tensor::TensorPtr>(inputs, i);
}

std::optional<StringImmPtr> ValueConverter::ToStringOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<StringImmPtr>(inputs, i);
}

std::optional<TypePtr> ValueConverter::ToDtypeOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<TypePtr>(inputs, i);
}

std::optional<ValueTuplePtr> ValueConverter::ToValueTupleOptional(const ValuePtrList &inputs, size_t i) {
  return ConvertOptional<ValueTuplePtr>(inputs, i);
}
}  // namespace mindspore::runtime
