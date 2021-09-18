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
#include <unordered_map>

#include "utils/visible.h"
#include "utils/compact_set.h"
#include "utils/hashing.h"
#include "ir/anf.h"

namespace mindspore::api {

class FuncGraph;
using FuncGraphPtr = std::shared_ptr<FuncGraph>;

class FuncGraphManager;
using FuncGraphManagerPtr = std::shared_ptr<FuncGraphManager>;

using AnfNodeIndexSet = CompactSet<std::pair<AnfNodePtr, int>>;
using NodeUsersMap = std::unordered_map<AnfNodePtr, AnfNodeIndexSet, PointerHash<AnfNodePtr>>;

class MS_CORE_API FuncGraphManager {
 public:
  FuncGraphManager() = default;
  virtual ~FuncGraphManager() = default;

  virtual bool Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) = 0;
  virtual void SetEdge(const AnfNodePtr &node, int index, const AnfNodePtr &value) = 0;
  virtual void AddEdge(const AnfNodePtr &node, const AnfNodePtr &value) = 0;
  virtual const NodeUsersMap &node_users() const = 0;

  static FuncGraphManagerPtr Manage(const FuncGraphPtr &func_graph, bool manage = true);
};

}  // namespace mindspore::api

#endif  // MINDSPORE_CORE_API_IR_FUNC_GRAPH_MANAGER_H_
