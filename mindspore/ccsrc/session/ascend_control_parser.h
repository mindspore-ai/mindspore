/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_SESSION_ASCEND_CONTROL_PARSER_H
#define MINDSPORE_CCSRC_SESSION_ASCEND_CONTROL_PARSER_H

#include <set>
#include <map>
#include <vector>
#include <tuple>
#include "session/kernel_graph.h"
#include "utils/base_ref.h"
#include "utils/contract.h"

namespace mindspore {
namespace session {
class AscendControlParser {
 public:
  static void ChildGraphDataAssign(const std::map<uint32_t, KernelGraphPtr> &graph_id_map);
  static void LinkGraph(NotNull<KernelGraphPtr> kg);

  static void InsertDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> attch_node);
  static void InsertControlDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> first_node,
                                         NotNull<AnfNodePtr> second_node);
  static void ExecutorValidate(NotNull<KernelGraphPtr> root_graph);
  static void UpdateChildGraphOrder(NotNull<KernelGraphPtr> kg);

 private:
  static NotNull<CNodePtr> ProcessKernelGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &last_node,
                                              const CNodePtr &last_label,
                                              const NotNull<std::set<KernelGraphPtr> *> memo);
  static void RecurseCall(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node, const CNodePtr &next_node,
                          const NotNull<std::set<KernelGraphPtr> *> memo);
  static void RecurseSwitch(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node, const CNodePtr &next_node,
                            const NotNull<std::set<KernelGraphPtr> *> memo);
  static void RecurseSwitchLayer(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node, const CNodePtr &next_node,
                                 const NotNull<std::set<KernelGraphPtr> *> memo);

  static void LinkParentGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &from_graph_call_node,
                              const CNodePtr &last_label);
  static std::tuple<CNodePtr, KernelGraphPtr> ParsePartial(NotNull<AnfNodePtr> node);

  static void InsertMultipleAssignToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> from, NotNull<AnfNodePtr> to);
  static void InsertAssignToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> from, NotNull<AnfNodePtr> to);

  // root graph order
  static bool CheckLabelIndex(uint32_t order_index, uint32_t label_index, const CNodePtr &cnode,
                              NotNull<KernelGraphPtr> graph);
  static std::vector<CNodePtr> RecurseGraph(NotNull<KernelGraphPtr> graph,
                                            const NotNull<std::set<KernelGraphPtr> *> memo);
};
}  // namespace session
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_SESSION_ASCEND_CONTROL_PARSER_H
