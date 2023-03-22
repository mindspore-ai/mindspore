/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ir/func_graph.h"
#include "utils/hash_map.h"

namespace mindspore::session {
class ExecOrderBuilder {
 public:
  ExecOrderBuilder() = default;

  ~ExecOrderBuilder();

  std::vector<CNodePtr> Build(FuncGraph *graph);

 private:
  void ClearLinkInfo();

  void BuildLinkInfo();

  void FindIndependentNodes();

  bool CanVisitInput(bool visit_with_refcount, const AnfNodePtr &input, mindspore::HashSet<AnfNodePtr> *visited);

  std::vector<CNodePtr> Build();

  void EnqueueReadyNodes(const AnfNodePtr &node, std::deque<AnfNodePtr> *visit_queue, bool comm_first = true);

  bool PrintLoopNodesIfExist(const AnfNodePtr &node, std::set<AnfNodePtr> *visited_nodes,
                             mindspore::HashMap<AnfNodePtr, AnfNodePtr> *next_nodes);

  void CheckLoop();

  bool IsTrivialNode(const AnfNodePtr &node);

  FuncGraph *graph_{nullptr};
  std::stack<AnfNodePtr> independent_nodes_;
  mindspore::HashMap<AnfNodePtr, size_t> node_input_num_;
  mindspore::HashMap<AnfNodePtr, size_t> node_output_num_;
  mindspore::HashMap<AnfNodePtr, std::vector<AnfNodePtr>> node_input_edges_;
  mindspore::HashMap<AnfNodePtr, std::vector<AnfNodePtr>> node_output_edges_;
  std::set<AnfNodePtr> trivial_nodes_;
};
}  // namespace mindspore::session
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SESSION_EXEC_ORDER_BUILDER_H
