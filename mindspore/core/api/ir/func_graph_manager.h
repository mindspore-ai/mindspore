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

#ifndef MINDSPORE_CORE_API_IR_FUNC_GRAPH_MANAGER_H_
#define MINDSPORE_CORE_API_IR_FUNC_GRAPH_MANAGER_H_

#include <memory>
#include <utility>

#include "utils/visible.h"
#include "utils/compact_set.h"
#include "utils/hash_map.h"
#include "utils/hashing.h"
#include "ir/anf.h"

namespace mindspore::deprecated::api {
class FuncGraph;
using FuncGraphPtr = std::shared_ptr<FuncGraph>;

class FuncGraphManager;
using FuncGraphManagerPtr = std::shared_ptr<FuncGraphManager>;

using AnfNodeIndexSet = CompactSet<std::pair<AnfNodePtr, int>>;
using NodeUsersMap = mindspore::HashMap<AnfNodePtr, AnfNodeIndexSet, PointerHash<AnfNodePtr>>;

/// \brief FuncGraphManager defines interface for function graph management.
class MS_CORE_API FuncGraphManager {
 public:
  /// \brief Constructor of FuncGraphManager.
  FuncGraphManager() = default;

  /// \brief Destructor of FuncGraphManager.
  virtual ~FuncGraphManager() = default;

  /// \brief Replace an old node with a new node, related edges are all updated.
  ///
  /// \param[in] old_node The old node to be replaced.
  /// \param[in] new_node The new node that replace the old one.
  ///
  /// \return True if the node is successfully replaced, false otherwise.
  virtual bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) = 0;

  /// \brief Change an existed edge by replace its input node.
  ///
  /// \param[in] node The output node of the edge.
  /// \param[in] index The input index in output node.
  /// \param[in] value The new input node of the edge.
  virtual void SetEdge(const AnfNodePtr &node, int index, const AnfNodePtr &value) = 0;

  /// \brief Adds a new edge between the given two nodes.
  ///
  /// \param[in] node The output node of the edge.
  /// \param[in] value The input node of the edge.
  virtual void AddEdge(const AnfNodePtr &node, const AnfNodePtr &value) = 0;

  /// \brief Get the node to users map.
  ///
  /// \return The node to users map.
  virtual const NodeUsersMap &node_users() const = 0;

  /// \brief Manage the give function graph.
  ///
  /// \param[in] func_graph The function graph to be managed.
  /// \param[in] manage If true, the created manager will be set in graph.
  ///
  /// \return The manager that manages the given function graph.
  static FuncGraphManagerPtr Manage(const FuncGraphPtr &func_graph, bool manage = true);
};
}  // namespace mindspore::deprecated::api

#ifndef USE_DEPRECATED_API
#define USE_DEPRECATED_API
namespace mindspore {
namespace api = deprecated::api;
}
#endif

#endif  // MINDSPORE_CORE_API_IR_FUNC_GRAPH_MANAGER_H_
