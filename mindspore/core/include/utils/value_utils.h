/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_VALUE_UTILS_H
#define MINDSPORE_CORE_UTILS_VALUE_UTILS_H

#include <vector>
#include <set>
#include <utility>
#include <string>
#include <optional>
#include <type_traits>

#include "ir/anf.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "abstract/abstract_value.h"
#include "mindapi/base/macros.h"

namespace mindspore {
// ArrayValue functions as a std::vector that verifies unknown values. ArrayValue uses std::vector<T> to hold the
// contents of the Sequence or Tensor flattened elements and provides an interface to determine whether each element is
// ValueAny.
template <typename T>
class ArrayValue {
 public:
  ArrayValue(std::vector<T> &&data, std::set<size_t> &&unknown_value_indexes)
      : data_(std::move(data)), unknown_value_indexes_(std::move(unknown_value_indexes)) {}

  ArrayValue(const ArrayValue &) = default;
  ArrayValue &operator=(const ArrayValue &) = default;

  ArrayValue(ArrayValue &&other) {
    data_ = std::move(other.data_);
    unknown_value_indexes_ = std::move(other.unknown_value_indexes_);
  }

  ArrayValue &operator=(ArrayValue &&other) {
    data_ = std::move(other.data_);
    unknown_value_indexes_ = std::move(other.unknown_value_indexes_);
    return *this;
  }

  ~ArrayValue() = default;

  // Access the value of Array at the index position.
  // Note: The value at position index can not be unknown, otherwise throw an exception.
  const T &operator[](size_t index) const {
    if (index >= data_.size()) {
      MS_LOG(EXCEPTION) << "The index[" << index << "] is out of range, element size is: " << data_.size();
    }
    if (IsValueUnknown(index)) {
      MS_LOG(EXCEPTION) << "Try to get unknown value.";
    }
    return data_[index];
  }

  // Verify that the value at position index in ArrayValue is unknown.
  bool IsValueUnknown(size_t index) const { return unknown_value_indexes_.find(index) != unknown_value_indexes_.end(); }

  // Verify whether exist unknown value in ArrayValue.
  bool HasUnknownValue() const { return !unknown_value_indexes_.empty(); }

  // Convert the ArrayValue to std::vector, only work when there is no unknown value in ArrayValue.
  const std::vector<T> &ToVector() const {
    if (HasUnknownValue()) {
      MS_LOG(EXCEPTION) << "Can not convert vector, there is unknown value in ArrayValue.";
    }
    return data_;
  }

  // Convert the ArrayValue to a string which contains all element in ArrayValue.
  std::string ToString() const {
    std::ostringstream oss;
    size_t element_size = size();
    oss << "{ ";
    for (size_t i = 0; i < element_size; i++) {
      oss << (!IsValueUnknown(i) ? std::to_string(data_[i]) : "ValueUnknown");
      if (i < element_size - 1) {
        oss << ", ";
      }
    }
    oss << " }";
    return oss.str();
  }

  // Get element number in ArrayValue.
  size_t size() const { return data_.size(); }

 private:
  // Use vector to hold the contents parsed from Sequence or Tensor Value.
  std::vector<T> data_;
  // Records the index whose value is unknown (ValueAny) in the data_ vector.
  std::set<size_t> unknown_value_indexes_;
};

// This interface is only used to convert values of type Sequence or Tensor to std::vector.
// Input can be AbstractTensor/AbstractSequence from frontend or KernelTensor from backend.
template <typename T>
MS_CORE_API std::optional<ArrayValue<T>> GetArrayValue(const abstract::AbstractBasePtr &abs_base);

template <typename T>
MS_CORE_API std::optional<ArrayValue<T>> GetArrayValue(const ValuePtr &value);

// This interface is only used to get value for scalar data.
template <typename T>
MS_CORE_API std::optional<T> GetScalarValue(const ValuePtr &value);

// ABI-safe interfaces: get scalar value through pointer (avoids cross-module std::optional issues)
template <typename T>
MS_CORE_API bool GetScalarValuePtr(const ValuePtr &value, T *out_value);

// Get the scalar/std::string value with check
template <typename T, typename std::enable_if<std::is_scalar<std::decay_t<T>>::value ||
                                              std::is_same_v<std::decay_t<T>, std::string>>::type * = nullptr>
T GetValueWithCheck(const ValuePtr &value) {
  auto opt = GetScalarValue<T>(value);
  if (!opt.has_value()) {
    MS_LOG(EXCEPTION) << "Get scalar or string value from " << value->ToString() << " with check failed.";
  }
  return opt.value();
}

// Template classes used to detect whether a type is a vector.
template <typename T>
struct IsVectorImpl : std::false_type {};
template <typename T>
struct IsVectorImpl<std::vector<T>> : std::true_type {};
template <typename T>
struct IsVector {
  static constexpr bool value = IsVectorImpl<std::decay_t<T>>::value;
};

// Get the std::vector value with check
template <typename T, typename std::enable_if<IsVector<T>::value>::type * = nullptr>
T GetValueWithCheck(const ValuePtr &value) {
  auto opt = GetArrayValue<typename T::value_type>(value);
  if (!opt.has_value()) {
    MS_LOG(EXCEPTION) << "Get array value from " << value->ToString() << " with check failed.";
  }
  return opt.value().ToVector();
}

template <typename T>
T GetScalarCastValue(const std::string &op_name, const ValuePtr &elem);

inline bool IsValueKnown(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  return !value->isa<ValueAny>() && !value->isa<None>();
}

inline bool IsValueKnown(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return IsValueKnown(abs->GetValue());
}

template <typename T>
MS_CORE_API T TensorItem(const tensor::TensorPtr &tensor);
}  //  namespace mindspore
#endif  //  MINDSPORE_CORE_UTILS_VALUE_UTILS_H
