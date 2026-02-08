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
#include <string>
#include <vector>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/backend/visible.h"

namespace mindspore::runtime {
class PYBOOST_API ValueConverter {
 public:
  template <typename T>
  static T Convert(const ValuePtr &input) {
    MS_EXCEPTION_IF_NULL(input);
    auto t = input->template cast<T>();
    if (t == nullptr) {
      MS_LOG(EXCEPTION) << "Get input type " << input->ToString() << ", but want to get " << typeid(T).name();
    }
    return t;
  }
  template <typename T, typename U>
  static U ConvertBasic(const ValuePtr &input) {
    MS_EXCEPTION_IF_NULL(input);
    auto t = input->template cast<T>();
    if (t == nullptr) {
      MS_LOG(EXCEPTION) << "Get input type " << input->ToString() << ", but want to get " << typeid(T).name();
    }
    return GetValue<U>(t);
  }
  static Int64ImmPtr ToInt(const ValuePtr &input);
  static FP32ImmPtr ToFloat(const ValuePtr &input);
  static BoolImmPtr ToBool(const ValuePtr &input);
  static ScalarPtr ToScalar(const ValuePtr &input);
  static tensor::TensorPtr ToTensor(const ValuePtr &input);
  static StringImmPtr ToString(const ValuePtr &input);
  static TypePtr ToDtype(const ValuePtr &input);
  static ValueTuplePtr ToValueTuple(const ValuePtr &input);

  static ValuePtr ToValue(const tensor::TensorPtr &tensor);
  static ValuePtr ToValue(const std::vector<tensor::TensorPtr> &tensors);
  static std::vector<int64_t> ToBasicIntVector(const ValuePtr &input);
  static int64_t ToBasicInt(const ValuePtr &input);

  template <typename T>
  static std::optional<T> ConvertOptional(const ValuePtr &input) {
    if (input->template isa<None>()) {
      return std::nullopt;
    }
    auto t = input->template cast<T>();
    MS_EXCEPTION_IF_NULL(t);
    return std::make_optional<T>(t);
  }

  template <typename T, typename U>
  static std::optional<U> ConvertBasicOptional(const ValuePtr &input) {
    if (input->template isa<None>()) {
      return std::nullopt;
    }
    auto t = input->template cast<T>();
    MS_EXCEPTION_IF_NULL(t);
    return std::make_optional<U>(GetValue<U>(t));
  }

  static std::optional<Int64ImmPtr> ToIntOptional(const ValuePtr &input);
  static std::optional<FP32ImmPtr> ToFloatOptional(const ValuePtr &input);
  static std::optional<BoolImmPtr> ToBoolOptional(const ValuePtr &input);
  static std::optional<ScalarPtr> ToScalarOptional(const ValuePtr &input);
  static std::optional<tensor::TensorPtr> ToTensorOptional(const ValuePtr &input);
  static std::optional<StringImmPtr> ToStringOptional(const ValuePtr &input);
  static std::optional<TypePtr> ToDtypeOptional(const ValuePtr &input);
  static std::optional<ValueTuplePtr> ToValueTupleOptional(const ValuePtr &input);

  static std::optional<std::vector<int64_t>> ToBasicIntVectorOptional(const ValuePtr &input);
  static std::optional<int64_t> ToBasicIntOptional(const ValuePtr &input);

  static tensor::TensorPtr ContiguousTensorValue(device::DeviceType device_target, const tensor::TensorPtr &tensor);
  static ValueTuplePtr ContiguousTensorValue(device::DeviceType device_target, const ValueTuplePtr &tuple);
  template <typename T>
  static std::optional<T> ContiguousTensorValue(device::DeviceType device_target, const std::optional<T> &val) {
    if (!val.has_value()) {
      return val;
    }
    return std::make_optional<T>(ContiguousTensorValue(device_target, val.value()));
  }
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_FUNCTION_VALUE_CONVERTER_H_
