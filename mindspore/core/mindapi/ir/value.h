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

#ifndef MINDSPORE_CORE_MINDAPI_IR_VALUE_H_
#define MINDSPORE_CORE_MINDAPI_IR_VALUE_H_

#include <vector>
#include <string>
#include <type_traits>
#include "mindapi/base/base.h"
#include "mindapi/ir/common.h"

namespace mindspore::api {
template <typename T>
struct ImmTrait {};

#define MIND_API_IMM_TRAIT(typeimm, prototype) \
  template <>                                  \
  struct ImmTrait<prototype> {                 \
    using type = SharedPtr<typeimm>;           \
  }

/// \brief Value represents a value in expression.
class MIND_API Value : public Base {
 public:
  MIND_API_BASE_MEMBER(Value);

  /// \brief Get the type of this Value.
  ///
  /// \return The type.
  TypePtr type() const;

  /// \brief Get the abstract of this Value.
  ///
  /// \return Abstract of this Value.
  AbstractBasePtr ToAbstract() const;
};

/// \brief ValueSequence represents a sequence of values.
class MIND_API ValueSequence : public Value {
 public:
  MIND_API_BASE_MEMBER(ValueSequence);

  /// \brief Get the size of this ValueSequence.
  ///
  /// \return The size as the number of elements.
  std::size_t size() const;

  /// \brief Get the list of values in this ValueSequence.
  ///
  /// \return The list of element values.
  std::vector<ValuePtr> value() const;
};

using ValueSequencePtr = SharedPtr<ValueSequence>;

/// \brief ValueTuple represents a value tuple.
class MIND_API ValueTuple : public ValueSequence {
 public:
  MIND_API_BASE_MEMBER(ValueTuple);

  /// \brief Constructor of ValueTuple.
  ///
  /// \param[in] elements The elements of the tuple.
  explicit ValueTuple(const std::vector<ValuePtr> &elements);
};

using ValueTuplePtr = SharedPtr<ValueTuple>;

/// \brief StringImm defines a Value whose type is string.
class MIND_API StringImm : public Value {
 public:
  MIND_API_BASE_MEMBER(StringImm);

  /// \brief Create StringImm with the given string.
  ///
  /// \param[in] str The given string value.
  explicit StringImm(const std::string &str);

  /// \brief Get the string value of this StringImm.
  ///
  /// \return The string value of this StringImm.
  const std::string &value() const;
};

using StringImmPtr = SharedPtr<StringImm>;

MIND_API_IMM_TRAIT(StringImm, std::string);

/// \brief Scalar defines interface for scalar data.
class MIND_API Scalar : public Value {
 public:
  MIND_API_BASE_MEMBER(Scalar);
};

/// \brief BoolImm defines interface for bool data.
class MIND_API BoolImm : public Scalar {
 public:
  MIND_API_BASE_MEMBER(BoolImm);

  /// \brief Create BoolImm with the given bool value.
  ///
  /// \param[in] b The given bool value.
  explicit BoolImm(bool b);

  /// \brief Get the bool value of this BoolImm.
  ///
  /// \return The bool value of this BoolImm.
  bool value() const;
};

using BoolImmPtr = SharedPtr<BoolImm>;

MIND_API_IMM_TRAIT(BoolImm, bool);

/// \brief IntegerImm defines interface for integer data.
class MIND_API IntegerImm : public Scalar {
 public:
  MIND_API_BASE_MEMBER(IntegerImm);
};

/// \brief Int8Imm defines interface for int8 data.
class MIND_API Int8Imm : public IntegerImm {
 public:
  MIND_API_BASE_MEMBER(Int8Imm);

  /// \brief Create Int8Imm with the given int8 value.
  ///
  /// \param[in] value The given int8 value.
  explicit Int8Imm(int8_t value);

  /// \brief Get the int8 value of this Int8Imm.
  ///
  /// \return The int8 value of this Int8Imm.
  int8_t value() const;
};

using Int8ImmPtr = SharedPtr<Int8Imm>;

MIND_API_IMM_TRAIT(Int8Imm, int8_t);

/// \brief Int16Imm defines interface for int16 data.
class MIND_API Int16Imm : public IntegerImm {
 public:
  MIND_API_BASE_MEMBER(Int16Imm);

  /// \brief Create Int1I6mm with the given int16 value.
  ///
  /// \param[in] value The given int16 value.
  explicit Int16Imm(int16_t value);

  /// \brief Get the int16 value of this Int16Imm.
  ///
  /// \return The int16 value of this Int16Imm.
  int16_t value() const;
};

using Int16ImmPtr = SharedPtr<Int16Imm>;

MIND_API_IMM_TRAIT(Int16Imm, int16_t);

/// \brief Int32Imm defines interface for int32 data.
class MIND_API Int32Imm : public IntegerImm {
 public:
  MIND_API_BASE_MEMBER(Int32Imm);

  /// \brief Create Int32Imm with the given int32 value.
  ///
  /// \param[in] value The given int32 value.
  explicit Int32Imm(int32_t value);

  /// \brief Get the int32 value of this Int32Imm.
  ///
  /// \return The int32 value of this Int32Imm.
  int32_t value() const;
};

using Int32ImmPtr = SharedPtr<Int32Imm>;

MIND_API_IMM_TRAIT(Int32Imm, int32_t);

/// \brief Int64Imm defines interface for int64 data.
class MIND_API Int64Imm : public IntegerImm {
 public:
  MIND_API_BASE_MEMBER(Int64Imm);

  /// \brief Create Int64Imm with the given int64 value.
  ///
  /// \param[in] value The given int64 value.
  explicit Int64Imm(int64_t value);

  /// \brief Get the int64 value of this Int64Imm.
  ///
  /// \return The int64 value of this Int64Imm.
  int64_t value() const;
};

using Int64ImmPtr = SharedPtr<Int64Imm>;

MIND_API_IMM_TRAIT(Int64Imm, int64_t);

/// \brief UInt8Imm defines interface for uint8 data.
class MIND_API UInt8Imm : public IntegerImm {
 public:
  MIND_API_BASE_MEMBER(UInt8Imm);

  /// \brief Create UInt8Imm with the given uint8 value.
  ///
  /// \param[in] value The given uint8 value.
  explicit UInt8Imm(uint8_t value);

  /// \brief Get the uint8 value of this UInt8Imm.
  ///
  /// \return The uint8 value of this UInt8Imm.
  uint8_t value() const;
};

using UInt8ImmPtr = SharedPtr<UInt8Imm>;

MIND_API_IMM_TRAIT(UInt8Imm, uint8_t);

/// \brief FloatImm defines interface for float data.
class MIND_API FloatImm : public Scalar {
 public:
  MIND_API_BASE_MEMBER(FloatImm);
};

/// \brief FP32Imm defines interface for float32 data.
class MIND_API FP32Imm : public FloatImm {
 public:
  MIND_API_BASE_MEMBER(FP32Imm);

  /// \brief Create FP32Imm with the given float value.
  ///
  /// \param[in] value The given float value.
  explicit FP32Imm(float value);

  /// \brief Get the float value of this FP32Imm.
  ///
  /// \return The float value of this FP32Imm.
  float value() const;
};

using FP32ImmPtr = SharedPtr<FP32Imm>;

MIND_API_IMM_TRAIT(FP32Imm, float);

/// \brief FP64Imm defines interface for float64 data.
class MIND_API FP64Imm : public FloatImm {
 public:
  MIND_API_BASE_MEMBER(FP64Imm);

  /// \brief Create FP64Imm with the given float value.
  ///
  /// \param[in] value The given float value.
  explicit FP64Imm(double value);

  /// \brief Get the float value of this FP64Imm.
  ///
  /// \return The float value of this FP64Imm.
  double value() const;
};

using FP64ImmPtr = SharedPtr<FP64Imm>;

MIND_API_IMM_TRAIT(FP64Imm, double);

// === Utility functions for Value === //

/// \brief Create a Value object from a primitive type value.
///
/// \param[in] v The primitive type value.
///
/// \return The created Value object with the given primitive type value.
template <typename T, typename U = typename ImmTrait<T>::type::element_type>
inline ValuePtr MakeValue(T v) {
  return MakeShared<U>(v);
}

/// \brief Create a StringImm Value object from a C string.
///
/// \param[in] s The C string.
///
/// \return The created StringImm Value object.
inline ValuePtr MakeValue(const char *s) { return MakeShared<StringImm>(std::string(s)); }

/// \brief Create an Int64Imm Value object from a int value.
///
/// \param[in] i The int value.
///
/// \return The created Int64Imm Value object.
inline ValuePtr MakeValue(int i) { return MakeShared<Int64Imm>(static_cast<int64_t>(i)); }

/// \brief Create a ValueSequence object from a vector of values.
///
/// \param[in] values The vector of values.
///
/// \return The created ValueSequence object.
inline ValuePtr MakeValue(const std::vector<ValuePtr> &values) { return MakeShared<ValueTuple>(values); }

/// \brief Create a ValueSequence object from a vector of primitive type values.
///
/// \param[in] values The vector of primitive values.
///
/// \return The created ValueSequence object.
template <typename T, typename = typename std::enable_if_t<is_vector<T>::value, T>>
inline ValuePtr MakeValue(const T &values) {
  std::vector<ValuePtr> value_vector;
  value_vector.reserve(values.size());
  for (auto value : values) {
    value_vector.emplace_back(MakeValue(value));
  }
  return MakeShared<ValueTuple>(value_vector);
}

/// \brief Get primitive type value from a Value object.
///
/// \param[in] value The pointer to the Value object.
///
/// \return The primitive type value of the Value object.
template <typename T, typename U = typename ImmTrait<T>::type>
inline T GetValue(const ValuePtr &value) {
  if (value == nullptr) {
    return T();
  }
  U imm = value->cast<U>();
  if (imm == nullptr) {
    return T();
  }
  return imm->value();
}

/// \brief Get element values from a ValueSequence object.
///
/// \param[in] value The pointer to the ValueSequence object.
///
/// \return The values as a vector, empty if the input is not a ValueSequence.
template <typename T, typename S = typename std::decay_t<T>,
          typename U = typename std::enable_if_t<is_vector<S>::value, typename S::value_type>>
std::vector<U> GetValue(const ValuePtr &value) {
  if (value == nullptr) {
    return {};
  }
  auto seq = value->cast<ValueSequencePtr>();
  if (seq == nullptr) {
    return {};
  }
  if constexpr (std::is_same_v<ValuePtr, U>) {
    return seq->value();
  } else {
    auto elements = seq->value();
    std::vector<U> result;
    result.reserve(elements.size());
    for (auto &e : elements) {
      result.emplace_back(GetValue<U>(e));
    }
    return result;
  }
}
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_VALUE_H_
