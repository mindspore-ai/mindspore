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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_FUNCTION_VALUE_CONVERTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_FUNCTION_VALUE_CONVERTER_H_

#include <optional>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/backend/visible.h"

namespace mindspore::runtime {
class BACKEND_EXPORT ValueConverter {
 public:
  template <typename T>
  static T Convert(const ValuePtrList &inputs, size_t i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    auto t = input->template cast<T>();
    MS_EXCEPTION_IF_NULL(t);
    return t;
  }
  static Int64ImmPtr ToInt(const ValuePtrList &inputs, size_t i);
  static FP32ImmPtr ToFloat(const ValuePtrList &inputs, size_t i);
  static BoolImmPtr ToBool(const ValuePtrList &inputs, size_t i);
  static ScalarPtr ToScalar(const ValuePtrList &inputs, size_t i);
  static tensor::TensorPtr ToTensor(const ValuePtrList &inputs, size_t i);
  static StringImmPtr ToString(const ValuePtrList &inputs, size_t i);
  static TypePtr ToDtype(const ValuePtrList &inputs, size_t i);
  static ValueTuplePtr ToValueTuple(const ValuePtrList &inputs, size_t i);

  template <typename T>
  static std::optional<T> ConvertOptional(const ValuePtrList &inputs, size_t i) {
    const auto &input = inputs[i];
    if (input->template isa<None>()) {
      return std::nullopt;
    }
    auto t = input->template cast<T>();
    MS_EXCEPTION_IF_NULL(t);
    return std::make_optional<T>(t);
  }
  static std::optional<Int64ImmPtr> ToIntOptional(const ValuePtrList &inputs, size_t i);
  static std::optional<FP32ImmPtr> ToFloatOptional(const ValuePtrList &inputs, size_t i);
  static std::optional<BoolImmPtr> ToBoolOptional(const ValuePtrList &inputs, size_t i);
  static std::optional<ScalarPtr> ToScalarOptional(const ValuePtrList &inputs, size_t i);
  static std::optional<tensor::TensorPtr> ToTensorOptional(const ValuePtrList &inputs, size_t i);
  static std::optional<StringImmPtr> ToStringOptional(const ValuePtrList &inputs, size_t i);
  static std::optional<TypePtr> ToDtypeOptional(const ValuePtrList &inputs, size_t i);
  static std::optional<ValueTuplePtr> ToValueTupleOptional(const ValuePtrList &inputs, size_t i);
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_FUNCTION_VALUE_CONVERTER_H_
