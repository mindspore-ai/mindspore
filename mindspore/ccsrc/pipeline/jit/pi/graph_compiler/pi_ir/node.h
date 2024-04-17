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
#ifndef MINDSPORE_PI_JIT_NODE_H_
#define MINDSPORE_PI_JIT_NODE_H_

#include <memory>
#include <limits>
#include <list>
#include <string>
#include <vector>
#include "pipeline/jit/pi/graph_compiler/pi_ir/debug_info.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/type.h"
#include "utils/hashing.h"

namespace mindspore {
namespace pijit {
namespace ir {
template <typename T>
struct is_shared_ptr : public std::false_type {};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : public std::true_type {};

/// \brief Node is a base class of all classes, which represent value, instruction, and so on.
class Node : public std::enable_shared_from_this<Node> {
 public:
  /**
   * \brief The constructor of Node.
   *
   * \return The instance of Node.
   */
  Node()
      : type_(std::make_shared<Type>()),
        node_id_(0),
        offset_(std::numeric_limits<size_t>::max()),
        debug_info_(std::make_shared<DebugInfo>("")) {}

  /// \brief Destructor.
  virtual ~Node() = default;

  /// \brief The description id of this class.
  static constexpr uint32_t kClassId = ConstStringHash("Node");

  /**
   * \brief Get type of this node.
   *
   * \return The type of this node.
   */
  const TypePtr &GetType() const { return type_; }

  /**
   * \brief Set type of this node.
   */
  void SetType(const TypePtr type) { type_ = type; }

  /**
   * \brief Get the id of this node.
   *
   * \return The id of this node.
   */
  size_t GetNodeId() const { return node_id_; }

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  virtual void SetNodeId(size_t *id) {
    node_id_ = *id;
    (*id)++;
  }

  /**
   * \brief Get the offset if this node is a instruction, else returns an invalid value.
   *
   * \return The offset of this node.
   */
  size_t GetOffset() const { return offset_; }

  /**
   * \brief Set the offset of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  virtual void SetOffset(size_t *offset) {
    if (IsOperation()) {
      if (NeedExtInstr()) {
        (*offset)++;
      }
      offset_ = *offset;
      (*offset)++;
    }
  }

  /**
   * \brief Sort all nodes, and give them a id and a offset if this node is a instruction.
   *
   * \note This method should only be called on the root function node
   */
  void Sort(size_t index = 0, size_t offset = 0) {
    SetNodeId(&index);
    SetOffset(&offset);
  }

  /**
   * \brief Get the debug information of this node.
   *
   * \return The debug information of this node.
   */
  const DebugInfoPtr &GetDebugInfo() const { return debug_info_; }

  /**
   * \brief Set the debug information of this node.
   *
   * \param[in] debug_info The debug information of this node.
   */
  void SetDebugInfo(const DebugInfoPtr &debug_info) { debug_info_ = debug_info; }

  /**
   * \brief Judge whether this class is derived from class with the given class id.
   *
   * \param[in] id Define a class id.
   *
   * \return The result of the judgment.
   */
  static bool IsDerivedFrom(uint32_t id) { return id == Node::kClassId; }

  /// \brief Judge whether this object is an instance of class with the given type id.
  ///
  /// \param[in] id Define a type id.
  ///
  /// \return The result of the judgment.
  virtual bool IsFromClass(uint32_t id) const { return Node::IsDerivedFrom(id); }

  /// \brief Get the id of this class.
  ///
  /// \return The id of this class.
  virtual uint32_t GetClassId() const { return Node::kClassId; }

  /**
   * \brief Judge whether the class id of this node is same as the given class id.
   *
   * \param[in] id Define a class id.
   *
   * \return The result of the judgment.
   */
  virtual bool IsSameClass(uint32_t id) const { return id == Node::kClassId; }

  /// \brief Get the name of this class.
  ///
  /// \return The node name.
  virtual std::string GetNodeName() const { return "Node"; }

  /**
   * \brief Judge whether this node is an instance of a given class which is derived from Node.
   *
   * \return The result of the judgment.
   */
  template <typename T,
            typename std::enable_if<!is_shared_ptr<T>::value && std::is_base_of<Node, T>::value, T>::type * = nullptr>
  inline bool isa() const {
    if constexpr (std::is_final<T>::value) {
      return this->IsSameClass(T::kClassId);
    } else {
      return this->IsFromClass(T::kClassId);
    }
  }

  /// \brief Cast a shared_ptr of this object to a given class.
  ///
  /// \return If success, a shared_ptr of the given class will be returned. Otherwise a nullptr will be returned.
  template <typename T, typename U = typename std::enable_if<is_shared_ptr<T>::value, typename T::element_type>::type>
  inline T cast() {
    if (isa<U>()) {
      return std::static_pointer_cast<U>(shared_from_this());
    }
    return nullptr;
  }

  /**
   * \brief Judge whether this node is an operation(instruction).
   *
   * \return The result of the judgment.
   */
  virtual bool IsOperation() const { return false; }

  /**
   * \brief Judge whether need to insert a EXTENDED_ARG instruction before this operation.
   *
   * \return The result of the judgment.
   */
  virtual bool NeedExtInstr() const { return false; }

  /**
   * \brief Mark whether this operation need to insert a EXTENDED_ARG instruction.
   *
   * \param[in] need the result.
   */
  virtual void SetNeedExtInstr(bool need) {}

  /**
   * \brief Get the description of this node.
   * \return The description.
   */
  virtual std::string ToString() const = 0;

 private:
  /// \brief The type of this node.
  TypePtr type_;
  /// \brief The id of this node, used to describe node when dump.
  size_t node_id_;
  /// \brief The offset of this node, only makes sense when the node is an operation.
  size_t offset_;
  /// \brief The debug information of this node.
  DebugInfoPtr debug_info_;
};

using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;

#define JIT_DECLARE_PARENT(current_t, parent_t)                                                                 \
  static constexpr uint32_t kClassId = ConstStringHash(#parent_t "_" #current_t);                               \
  static bool IsDerivedFrom(uint32_t id) { return (id == current_t::kClassId) || parent_t::IsDerivedFrom(id); } \
  bool IsFromClass(uint32_t id) const override { return current_t::IsDerivedFrom(id); }                         \
  bool IsSameClass(uint32_t id) const override { return id == current_t::kClassId; }                            \
  uint32_t GetClassId() const override { return current_t::kClassId; }                                          \
  std::string GetNodeName() const override { return #current_t; }
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_NODE_H_
