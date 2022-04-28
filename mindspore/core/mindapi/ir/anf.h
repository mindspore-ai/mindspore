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

#ifndef MINDSPORE_CORE_MINDAPI_IR_ANF_H_
#define MINDSPORE_CORE_MINDAPI_IR_ANF_H_

#include <vector>
#include <string>
#include "mindapi/base/base.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/abstract.h"
#include "mindapi/ir/primitive.h"
#include "mindapi/ir/value.h"

namespace mindspore::api {
/// \brief AnfNode is the basic class of the IR graph node.
class MIND_API AnfNode : public Base {
 public:
  MIND_API_BASE_MEMBER(AnfNode);

  /// \brief Obtain detailed information about scope namespace.
  ///
  /// \return Detailed information about scope namespace.
  std::string fullname_with_scope() const;

  /// \brief Obtain the inferred abstract value of this AnfNode.
  ///
  /// \return The inferred abstract value.
  AbstractBasePtr abstract() const;

  /// \brief Set the abstract value of this AnfNode.
  ///
  /// \param[in] abs New abstract value.
  void set_abstract(const AbstractBasePtr &abs);
};

/// \brief CNode represents a compute node with a set of input nodes.
class MIND_API CNode : public AnfNode {
 public:
  MIND_API_BASE_MEMBER(CNode);

  /// \brief Get the number of inputs.
  ///
  /// \return The number of inputs in this CNode.
  size_t size() const;

  /// \brief Get the input node of the given index.
  ///
  /// \param[in] i The given index.
  ///
  /// \return The input node of the given index.
  AnfNodePtr input(size_t i) const;

  /// \brief Get the input nodes.
  ///
  /// \return The input nodes of this CNode.
  std::vector<AnfNodePtr> inputs() const;

  /// \brief Set the input nodes for this CNode.
  ///
  /// \param[in] inputs Input nodes.
  void set_inputs(const std::vector<AnfNodePtr> &inputs);

  /// \brief Add an input node to this CNode.
  ///
  /// \param[in] input the input node to be added.
  void add_input(const AnfNodePtr &input);

  /// \brief Set fullname_with_scope for this CNode.
  ///
  /// \param[in] full_name The fullname_with_scope.
  void set_fullname_with_scope(const std::string &full_name);

  /// \brief Add a new attribute to this CNode.
  ///
  /// \param[in] name The name of the new attribute.
  /// \param[in] attr The value of the new attribute.
  void AddAttr(const std::string &name, const ValuePtr &attr);

  /// \brief Erase the attribute with the given name.
  ///
  /// \param[in] name The name of attribute.
  void EraseAttr(const std::string &name);

  /// \brief Get the attribute with the given name.
  ///
  /// \param[in] name The name of attribute.
  /// \return Attribute.
  ValuePtr GetAttr(const std::string &name) const;
};

using CNodePtr = SharedPtr<CNode>;

/// \brief Parameter represents the parameter inputs of a function.
class MIND_API Parameter : public AnfNode {
 public:
  MIND_API_BASE_MEMBER(Parameter);

  /// \brief Get the name of this Parameter.
  ///
  /// \return The name.
  std::string name() const;

  /// \brief Set the name of this Parameter.
  ///
  /// \param[in] name The name.
  void set_name(const std::string &name);

  /// \brief Check if there is a default parameter.
  ///
  /// \return True if this Parameter has a default parameter, otherwise false.
  bool has_default() const;

  /// \brief Set the default parameter.
  ///
  /// \param[in] param The default parameter.
  void set_default_param(const ValuePtr &param);

  /// \brief Get the default parameter.
  ///
  /// \return The default parameter.
  ValuePtr default_param() const;
};

using ParameterPtr = SharedPtr<Parameter>;

/// \brief ValueNode is a graph node that hold a value.
class MIND_API ValueNode : public AnfNode {
 public:
  MIND_API_BASE_MEMBER(ValueNode);

  /// \brief Create ValueNode with the given value.
  ///
  /// \param[in] value The value of this ValueNode.
  explicit ValueNode(const ValuePtr &value);

  /// \brief Get the value of this ValueNode.
  ///
  /// \return The value.
  ValuePtr value() const;
};

using ValueNodePtr = SharedPtr<ValueNode>;

// === ANF utility functions === //

/// \brief Create a ValueNode with the given value.
///
/// \param[in] value The given value.
///
/// \return The created ValueNode.
template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Value, T>, T>>
inline ValueNodePtr NewValueNode(const SharedPtr<T> &value) {
  return MakeShared<ValueNode>(value);
}

/// \brief Create a ValueNode with the given primitive type value.
///
/// \param[in] value The given primitive type value.
///
/// \return The created ValueNode.
template <typename T>
inline ValueNodePtr NewValueNode(T value) {
  return NewValueNode(MakeValue(value));
}

/// \brief Get the value from a node if it is a ValueNode.
///
/// \param[in] node The node which may hold a value.
///
/// \return A pointer to the value, nullptr if the node is not a ValueNode, or value not set.
inline ValuePtr GetValueNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  return value_node->value();
}

/// \brief Get the value with the given type from a node if it is a ValueNode.
///
/// \param[in] node The node which may hold a value.
///
/// \return A pointer to the value, nullptr if the node is not a ValueNode, or value not set, or value type is mismatch.
template <typename T, typename = typename std::enable_if_t<
                        is_wrapper_ptr<T>::value && std::is_base_of_v<Value, typename T::element_type>, T>>
inline T GetValueNode(const AnfNodePtr &node) {
  auto value = GetValueNode(node);
  if (value == nullptr) {
    return nullptr;
  }
  return value->cast<T>();
}

/// \brief Check whether the given node is a cnode with the given Primitive as the first input.
///
/// \param[in] node The given node to be checked.
/// \param[in] prim The Primitive value, nullptr means match any Primitive.
///
/// \return True if the node is cnode and the first input is the given Primitive, false otherwise.
MIND_API bool IsPrimitiveCNode(const AnfNodePtr &node, const PrimitivePtr &prim = nullptr);

/// \brief Check whether the given node is a ValueNode with the given Primitive.
///
/// \param[in] node The given node to be checked.
/// \param[in] prim The Primitive value.
///
/// \return True if the given node is a ValueNode with the given Primitive, false otherwise.
MIND_API bool IsPrimitive(const AnfNodePtr &node, const PrimitivePtr &prim);

/// \brief Check if a node is a data node.
/// Some nodes may be used internally to pass some non-data states, those nodes are not data nodes.
///
/// \param[in] node The node to be checked.
///
/// \return True if the node is a data node, false otherwise.
MIND_API bool IsDataNode(const AnfNodePtr &node);
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_ANF_H_
