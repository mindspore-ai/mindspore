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

#ifndef MINDSPORE_CORE_MINDAPI_IR_TYPE_H_
#define MINDSPORE_CORE_MINDAPI_IR_TYPE_H_

#include "mindapi/base/base.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/value.h"

namespace mindspore::api {
/// \brief Type defines the type of a value.
class MIND_API Type : public Value {
 public:
  MIND_API_BASE_MEMBER(Type);

  /// \brief Get the id of the Type object.
  ///
  /// \return The id of the Type object.
  TypeId type_id() const;

  /// \brief Get the number type of the Type object.
  ///
  /// \return The number type of this Type object, kTypeUnknown if this is not a number type.
  TypeId number_type() const;

  /// \brief Get the Type according to a TypeId.
  ///
  /// \param[in] id The id of the type.
  ///
  /// \return The pointer to the Type.
  static TypePtr GetType(TypeId id);

  /// \brief Get data size in bytes for the type according to a TypeId.
  ///
  /// \param[in] id The id of the type.
  ///
  /// \return The data size in bytes for the Type.
  static size_t GetSize(TypeId id);
};

/// \brief TensorType defines the type of a tensor.
class MIND_API TensorType : public Type {
 public:
  MIND_API_BASE_MEMBER(TensorType);

  /// \brief Construct TensorType from the given element type.
  ///
  /// \param[in] element_type The element type of the TensorType.
  explicit TensorType(const TypePtr &element_type);

  /// \brief Get the element type of this TensorType.
  ///
  /// \return The element type of this TensorType.
  TypePtr element() const;
};

using TensorTypePtr = SharedPtr<TensorType>;
}  // namespace mindspore::api

#endif  // MINDSPORE_CORE_MINDAPI_IR_TYPE_H_
