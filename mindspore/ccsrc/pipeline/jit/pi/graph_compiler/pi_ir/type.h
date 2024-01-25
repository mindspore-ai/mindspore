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
#ifndef MINDSPORE_PI_JIT_TYPE_H_
#define MINDSPORE_PI_JIT_TYPE_H_

#include <memory>
#include <string>

namespace mindspore {
namespace pijit {
namespace ir {
using TypeId = int;

class Type : public std::enable_shared_from_this<Type> {
 public:
  /// \brief The default constructor for Type.
  Type() : Type(0, "Unknown") {}

  /// \brief The constructor for Type.
  explicit Type(const TypeId type_id, const std::string &name) : type_id_(type_id), name_(name) {}

  /// \brief The copy constructor of Type.
  ///
  /// \param[in] other Define another instance of Type.
  ///
  /// \return The instance of Type.
  explicit Type(const Type &other) : std::enable_shared_from_this<Type>(other) {}

  /// \brief Destructor.
  virtual ~Type() = default;

  /// \brief The operator overloading for "==".
  ///
  /// \param[in] rhs Define the right operand of "==".
  ///
  /// \return The comparison result.
  virtual bool operator==(const Type &rhs) { return this == &rhs || (type_id_ == rhs.type_id_ && name_ == rhs.name_); }

  /// \brief Get the type id of this object.
  ///
  /// \return The type id.
  TypeId GetTypeId() const { return type_id_; }

  /// \brief Get the type name of this object.
  ///
  /// \return The type name.
  const std::string &GetName() const { return name_; }

  /// \brief Get the string representation of this object.
  ///
  /// \return The string representation.
  virtual std::string ToString() const { return GetName(); }

 private:
  /// \brief The id of this Type.
  TypeId type_id_;
  /// \brief The name of this Type.
  std::string name_;
};

using TypePtr = std::shared_ptr<Type>;
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_TYPE_H_
