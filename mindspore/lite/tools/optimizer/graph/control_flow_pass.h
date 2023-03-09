/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CONTROL_FLOW_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_CONTROL_FLOW_PASS_H_
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <set>
#include "schema/inner/model_generated.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore::opt {
class ControlFlowPass : public Pass {
 public:
  ControlFlowPass() : Pass("control_flow_pass") {}
  ~ControlFlowPass() override = default;
  bool Run(const FuncGraphPtr &fg) override;

 private:
  void ReplaceNode(const FuncGraphPtr &fg, const std::unordered_map<AnfNodePtr, AnfNodePtr> &replace_pairs);
  void VisitedNodesUsedByAfterParts(const std::set<AnfNodePtr> &visited_nodes,
                                    const std::vector<AnfNodePtr> &remain_nodes,
                                    std::vector<AnfNodePtr> *visited_nodes_used_by_after_fg);
  int SplitGraph(const FuncGraphPtr &fg, AnfNodePtr *control_flow_node, std::set<AnfNodePtr> *visited_nodes,
                 std::vector<AnfNodePtr> *remain_nodes);
  size_t GetItemVisitedNums(const std::set<AnfNodePtr> &visited_nodes, const AnfNodePtr &tuple_node);
  void MoveGetItemToVisited(const size_t &need_size, const AnfNodePtr &tuple_node, std::set<AnfNodePtr> *visited_nodes,
                            std::vector<AnfNodePtr> *remain_nodes);
  void BindGetItemNodes(std::set<AnfNodePtr> *visited_nodes, std::vector<AnfNodePtr> *remain_nodes);
  int CreateAfterGraph(const FuncGraphPtr &main_fg, const std::vector<AnfNodePtr> &remain_nodes,
                       const CNodePtr &aim_cnode, FuncGraphPtr *after_fg);

  // process while
  int CreateWhileCondCallNode(
    const FuncGraphPtr &fg, const CNodePtr &while_cnode, const std::vector<AnfNodePtr> &visited_nodes_used_by_after_fg,
    CNodePtr *cond_partial_cnode, std::vector<AnfNodePtr> *cond_nodes_used_by_after_partial,
    std::unordered_map<AnfNodePtr, AnfNodePtr> *visited_nodes_and_cond_fg_inputs_replace_pairs);
  int CreateWhileBodyPartialNode(const FuncGraphPtr &cond_fg, const CNodePtr &while_cnode, CNodePtr *body_partial_node);
  int CreateWhileAfterPartialNode(
    const FuncGraphPtr &main_fg, const FuncGraphPtr &cond_fg, const std::vector<AnfNodePtr> &remain_nodes,
    const std::vector<AnfNodePtr> &cond_nodes_used_by_after_partial,
    const std::unordered_map<AnfNodePtr, AnfNodePtr> &visited_nodes_and_cond_fg_inputs_replace_pairs,
    const CNodePtr *while_cnode, CNodePtr *after_partial_cnode);
  int ProcessWhileOp(const FuncGraphPtr &fg, const std::set<AnfNodePtr> &visited_nodes,
                     const std::vector<AnfNodePtr> &remain_nodes, const AnfNodePtr &while_node);

  // process if
  int CreateIfPartialNodeExternalInputs(const CNodePtr &if_cnode, const FuncGraphPtr &partial_fg,
                                        std::vector<AnfNodePtr> *then_partial_cnode_inputs);
  int CreateIfPartialNode(const FuncGraphPtr &fg, const size_t &index,
                          std::vector<AnfNodePtr> *fg_inputs_only_used_by_after_partial, const CNodePtr &if_cnode,
                          const FuncGraphPtr &after_fg, CNodePtr *then_partial_cnode);
  int CreateIfThenPartialNode(const FuncGraphPtr &main_fg,
                              std::vector<AnfNodePtr> *fg_inputs_only_used_by_after_partial, const CNodePtr &if_cnode,
                              const FuncGraphPtr &after_fg, CNodePtr *then_partial_cnode);
  int CreateIfElsePartialNode(const FuncGraphPtr &main_fg,
                              std::vector<AnfNodePtr> *fg_inputs_only_used_by_after_partial, const CNodePtr &if_cnode,
                              const FuncGraphPtr &after_fg, CNodePtr *else_partial_cnode);
  int ProcessIfOp(const FuncGraphPtr &fg, const std::set<AnfNodePtr> &visited_nodes,
                  const std::vector<AnfNodePtr> &remain_nodes, const AnfNodePtr &if_node);

  int ProcessControlOp(const FuncGraphPtr &fg);

  const size_t kCNodePrimIndex = 0;
  const size_t kCNodeFirstInputIndex = 1;
  const size_t kCNodeSecondInputIndex = 2;

  const size_t kGetItemInputSize = 3;
  const size_t kPartialFirstInputSize = 2;

  const size_t kWhileMinInputSize = 3;
  const size_t kWhileCondIndex = 1;
  const size_t kWhileBodyIndex = 2;

  const size_t kIfMinInputSize = 4;
  const size_t kIfThenIndex = 1;
  const size_t kIfElseIndex = 2;
  const size_t kIfCondIndex = 3;

  std::deque<FuncGraphPtr> to_process_q{};
};
}  // namespace mindspore::opt
#endif
