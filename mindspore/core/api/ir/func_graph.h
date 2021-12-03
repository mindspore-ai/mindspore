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

#ifndef MINDSPORE_CORE_API_IR_FUNC_GRAPH_H_
#define MINDSPORE_CORE_API_IR_FUNC_GRAPH_H_

#include <vector>
#include <memory>
#include <string>

#include "utils/visible.h"
#include "api/ir/func_graph_manager.h"

namespace mindspore::deprecated::api {
/// \brief FuncGraph defines interface for a function graph.
class MS_CORE_API FuncGraph {
 public:
  /// \brief Constructor of FuncGraph.
  FuncGraph() = default;

  /// \brief Destructor of FuncGraph.
  virtual ~FuncGraph() = default;

  /// \brief Get the input parameters.
  ///
  /// \return Input parameters of this graph.
  virtual const std::vector<AnfNodePtr> get_inputs() const = 0;

  /// \brief Get all parameters.
  ///
  /// \return All parameters of this graph.
  virtual const std::vector<AnfNodePtr> &parameters() const = 0;

  /// \brief Adds a parameter to this graph.
  ///
  /// \param[in] p The parameter to be added.
  virtual void add_parameter(const ParameterPtr &p) = 0;

  /// \brief Adds a new parameter to this graph.
  ///
  /// \return The new added parameter.
  virtual ParameterPtr add_parameter() = 0;

  /// \brief Get the output node.
  ///
  /// \return The output node, nullptr if output not set.
  virtual AnfNodePtr output() const = 0;

  /// \brief Get the return CNode.
  ///
  /// \return The return CNode, nullptr if no return node.
  virtual CNodePtr get_return() const = 0;

  /// \brief Set the output node.
  ///
  /// \param[in] value The output node to be set.
  /// \param[in] force_new_ret If true, a new return node is always created.
  virtual void set_output(const AnfNodePtr &value, bool force_new_ret = false) = 0;

  /// \brief Set the return node.
  ///
  /// \param[in] cnode The return CNode to be set.
  virtual void set_return(const CNodePtr &cnode) = 0;

  /// \brief Creates a new CNode in this graph.
  ///
  /// \param[in] inputs The input nodes of the new CNode.
  ///
  /// \return The created CNode.
  virtual CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs = std::vector<AnfNodePtr>()) = 0;

  /// \brief Creates a new primitive CNode in this graph.
  ///
  /// \param[in] primitive The primitive of the new CNode.
  /// \param[in] prim_inputs The argument inputs of the primitive CNode.
  ///
  /// \return The created primitive CNode.
  virtual CNodePtr NewCNode(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &prim_inputs) = 0;

  /// \brief Get all nodes in this graph.
  ///
  /// \return All nodes in this graph.
  virtual const AnfNodeSet &nodes() const = 0;

  /// \brief Check whether an attribute is set for this graph.
  ///
  /// \param[in] key The attribute key (name).
  ///
  /// \return True if the attribute with the given key is set, false otherwise.
  virtual bool has_attr(const std::string &key) const = 0;

  /// \brief Get an attribute value by its key.
  ///
  /// \param[in] key The attribute key (name).
  ///
  /// \return The attribute value for the given key, nullptr if attribute not found.
  virtual ValuePtr get_attr(const std::string &key) const = 0;

  /// \brief Set an attribute value.
  ///
  /// \param[in] key The attribute key (name).
  /// \param[in] value The attribute value.
  virtual void set_attr(const std::string &key, const ValuePtr &value) = 0;

  /// \brief Get the manager for this graph.
  ///
  /// \return The manager of this graph, nullptr if not set.
  virtual FuncGraphManagerPtr get_manager() const = 0;

  /// \brief Topological sort a graph from the given end node.
  ///
  /// \param[in] node The end node of the graph to be sorted.
  ///
  /// \return The sorted nodes.
  static std::vector<AnfNodePtr> TopoSort(const AnfNodePtr &node);

  /// \brief Creates an empty function graph.
  ///
  /// \return The created graph.
  static FuncGraphPtr Create();

  /// \brief Creates a value node that holds the given function graph.
  ///
  /// \param[in] func_graph The given function graph.
  //
  /// \return The created value node that holds the given function graph.
  static AnfNodePtr MakeValueNode(const FuncGraphPtr &func_graph);

  /// \brief Get the function graph from the input node.
  ///
  /// \param[in] input The input node.
  //
  /// \return The function graph if the input is value node that holds the graph, nullptr otherwise.
  static FuncGraphPtr GetFuncGraphFromAnfNode(const AnfNodePtr &input);
};

#ifndef USE_DEPRECATED_API
#define USE_DEPRECATED_API
namespace mindspore {
namespace api = deprecated::api;
}
#endif
}  // namespace mindspore::deprecated::api
#endif  // MINDSPORE_CORE_API_IR_FUNC_GRAPH_H_
