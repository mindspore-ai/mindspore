/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_SESSION_EXEC_ORDER_BUILDER_H
#define MINDSPORE_CCSRC_BACKEND_COMMON_SESSION_EXEC_ORDER_BUILDER_H

#include <vector>
#include <stack>
#include <queue>
#include <deque>
#include <set>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"

namespace mindspore::session {

using NodeUser = mindspore::HashMap<AnfNodePtr, std::vector<AnfNodePtr>>;
class ExecOrderBuilder {
 public:
  ExecOrderBuilder() = default;

  ~ExecOrderBuilder();

  void Build(KernelGraph *graph, std::vector<CNodePtr> *execution_order, NodeUser *node_user);

 private:
  void ClearLinkInfo();

  void BuildLinkInfo();

  void FindIndependentNodes();

  bool CanVisitInput(bool visit_with_refcount, const AnfNodePtr &input, SeenNum seen);

  void BuildByBFS();
  void BuildByDFS();

  void EnqueEagerDepend(const AnfNodePtr &node, std::deque<AnfNodePtr> *visit_queue);

  void EnqueueReadyNodes(const AnfNodePtr &node, std::deque<AnfNodePtr> *visit_queue, bool comm_first = true);

  bool PrintLoopNodesIfExist(const AnfNodePtr &node, std::set<AnfNodePtr> *visited_nodes,
                             mindspore::HashMap<AnfNodePtr, AnfNodePtr> *next_nodes);

  void CheckLoop();

  bool IsTrivialNode(const AnfNodePtr &node);
  void GetTrivialInputNode(const AnfNodePtr &node, SeenNum seen);

  std::vector<CNodePtr> *execution_order_{nullptr};
  KernelGraph *graph_{nullptr};
  std::stack<AnfNodePtr> independent_nodes_;
  mindspore::HashMap<AnfNodePtr, size_t> node_input_num_;
  mindspore::HashMap<AnfNodePtr, size_t> node_output_num_;
  mindspore::HashSet<AnfNodePtr> eager_visited_nodes_;
  NodeUser node_input_edges_;
  NodeUser *node_output_edges_{nullptr};
  mindspore::HashMap<AnfNodePtr, bool> trivial_nodes_;
};
}  // namespace mindspore::session
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SESSION_EXEC_ORDER_BUILDER_H
