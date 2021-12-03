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

#ifndef MINDSPORE_CORE_MINDAPI_IR_FUNC_GRAPH_H_
#define MINDSPORE_CORE_MINDAPI_IR_FUNC_GRAPH_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "mindapi/base/base.h"
#include "mindapi/ir/common.h"
#include "mindapi/ir/anf.h"
#include "mindapi/ir/primitive.h"
#include "mindapi/ir/value.h"
#include "mindapi/ir/utils.h"

namespace mindspore {
class FuncGraphManager;
}

namespace mindspore::api {
/// \brief FuncGraph defines interface for a function graph.
class MIND_API FuncGraph : public Value {
 public:
  MIND_API_BASE_MEMBER(FuncGraph);

  /// \brief Get the input parameters.
  ///
  /// \return Input parameters of this graph.
  std::vector<AnfNodePtr> get_inputs() const;

  /// \brief Get all parameters.
  ///
  /// \return All parameters of this graph.
  std::vector<AnfNodePtr> parameters() const;

  /// \brief Adds a parameter to this graph.
  ///
  /// \param[in] p The parameter to be added.
  void add_parameter(const ParameterPtr &p);

  /// \brief Adds a new parameter to this graph.
  ///
  /// \return The new added parameter.
  ParameterPtr add_parameter();

  /// \brief Get the output node.
  ///
  /// \return The output node, nullptr if output not set.
  AnfNodePtr output() const;

  /// \brief Get the return CNode.
  ///
  /// \return The return CNode, nullptr if no return node.
  CNodePtr get_return() const;

  /// \brief Set the output node.
  ///
  /// \param[in] value The output node to be set.
  /// \param[in] force_new_ret If true, a new return node is always created.
  void set_output(const AnfNodePtr &value, bool force_new_ret = false);

  /// \brief Set the return node.
  ///
  /// \param[in] cnode The return CNode to be set.
  void set_return(const CNodePtr &cnode);

  /// \brief Creates a new CNode in this graph.
  ///
  /// \param[in] inputs The input nodes of the new CNode.
  ///
  /// \return The created CNode.
  CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs = std::vector<AnfNodePtr>());

  /// \brief Creates a new primitive CNode in this graph.
  ///
  /// \param[in] primitive The primitive of the new CNode.
  /// \param[in] prim_inputs The argument inputs of the primitive CNode.
  ///
  /// \return The created primitive CNode.
  CNodePtr NewCNode(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &prim_inputs);

  /// \brief Get all nodes in this graph.
  ///
  /// \return All nodes in this graph.
  std::vector<AnfNodePtr> nodes() const;

  /// \brief Check whether an attribute is set for this graph.
  ///
  /// \param[in] key The attribute key (name).
  ///
  /// \return True if the attribute with the given key is set, false otherwise.
  bool has_attr(const std::string &key) const;

  /// \brief Get an attribute value by its key.
  ///
  /// \param[in] key The attribute key (name).
  ///
  /// \return The attribute value for the given key, nullptr if attribute not found.
  ValuePtr get_attr(const std::string &key) const;

  /// \brief Set an attribute value.
  ///
  /// \param[in] key The attribute key (name).
  /// \param[in] value The attribute value.
  void set_attr(const std::string &key, const ValuePtr &value);

  /// \brief Get the manager for this graph.
  ///
  /// \return The manager of this graph, nullptr if not set.
  FuncGraphManagerPtr manager() const;

  /// \brief Creates an empty function graph.
  ///
  /// \return The created function graph.
  static FuncGraphPtr Create();

  /// \brief Topological sort a graph from the given end node.
  ///
  /// \param[in] node The end node of the graph to be sorted.
  ///
  /// \return The sorted nodes.
  static std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &node);
};

/// \brief FuncGraphManager defines interface for function graph management.
class MIND_API FuncGraphManager {
 public:
  /// \brief Create FuncGraphManager with the given implementor object.
  ///
  /// \param[in] impl The pointer to the implementor object.
  explicit FuncGraphManager(const std::shared_ptr<mindspore::FuncGraphManager> &impl);

  /// \brief Get the shared_ptr to the underly implementation object.
  ///
  /// \return The shared_ptr to the underly implementation object.
  const std::shared_ptr<mindspore::FuncGraphManager> &impl() const { return impl_; }

  /// \brief Replace an old node with a new node, related edges are all updated.
  ///
  /// \param[in] old_node The old node to be replaced.
  /// \param[in] new_node The new node that replace the old one.
  ///
  /// \return True if the node is successfully replaced, false otherwise.
  bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node);

  /// \brief Change an existed edge by replace its input node.
  ///
  /// \param[in] node The output node of the edge.
  /// \param[in] index The input index in output node.
  /// \param[in] value The new input node of the edge.
  void SetEdge(const AnfNodePtr &node, int index, const AnfNodePtr &value);

  /// \brief Adds a new edge between the given two nodes.
  ///
  /// \param[in] node The output node of the edge.
  /// \param[in] value The input node of the edge.
  void AddEdge(const AnfNodePtr &node, const AnfNodePtr &value);

  /// \brief Find users of the given node.
  ///
  /// \param[in] node The node.
  ///
  /// \return Users of the given node, empty if user not found.
  std::vector<std::pair<AnfNodePtr, int>> GetUsers(const AnfNodePtr &node) const;

  /// \brief Manage the give function graph.
  ///
  /// \param[in] func_graph The function graph to be managed.
  /// \param[in] manage If true, the created manager will be set in the graph.
  ///
  /// \return The manager that manages the given function graph.
  static FuncGraphManagerPtr Manage(const FuncGraphPtr &func_graph, bool manage = true);

 private:
  const std::shared_ptr<mindspore::FuncGraphManager> impl_;
};
}  // namespace mindspore::api
#endif  // MINDSPORE_CORE_MINDAPI_IR_FUNC_GRAPH_H_
