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

#ifndef MINDSPORE_CORE_MINDAPI_IR_PRIMITIVE_H_
#define MINDSPORE_CORE_MINDAPI_IR_PRIMITIVE_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "mindapi/base/base.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/value.h"

namespace mindspore::api {
/// \brief Primitive defines a primitive operator.
class MIND_API Primitive : public Value {
 public:
  MIND_API_BASE_MEMBER(Primitive);

  /// \brief Create primitive with the given name.
  ///
  /// \param[in] name The primitive name.
  explicit Primitive(const std::string &name);

  /// \brief Get name of the primitive.
  ///
  /// \return The name of primitive.
  const std::string &name() const;

  /// \brief Add attribute to primitive.
  ///
  /// \param[in] name The attribute name.
  /// \param[in] attr The attribute value.
  /// \return The primitive to which attribute has been added.
  Primitive &AddAttr(const std::string &name, const ValuePtr &attr);

  /// \brief Add attributes by using a map, all elements of the map will be added to this primitive.
  ///
  /// \param[in] attrs The attribute map needs to be added in the primitive attribute.
  /// \return The primitive to which attribute has been added.
  Primitive &SetAttrs(const std::unordered_map<std::string, ValuePtr> &attrs);

  /// \brief Erase attribute to the primitive attribute map.
  ///
  /// \param[in] name The attribute name.
  void EraseAttr(const std::string &name);

  /// \brief Get attribute value by name.
  ///
  /// \param[in] name the attribute name.
  /// \return The value of the attribute, null if attribute name not found.
  ValuePtr GetAttr(const std::string &name) const;

  /// \brief Check If Primitive has an attribute with then given name.
  ///
  /// \param[in] name The attribute name.
  /// \return True if there is an attribute with the given name, otherwise false.
  bool HasAttr(const std::string &name) const;

  /// \brief Get all attributes of this primitive as a map.
  ///
  /// \return The attribute map of this primitive.
  std::unordered_map<std::string, ValuePtr> attrs() const;
};
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_PRIMITIVE_H_
