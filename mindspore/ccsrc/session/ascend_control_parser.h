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
#include <vector>
#include <tuple>
#include "session/kernel_graph.h"
#include "utils/base_ref.h"
#include "utils/contract.h"

namespace mindspore {
namespace session {

class AscendControlParser {
 public:
  static void LinkGraph(NotNull<KernelGraphPtr> kg);

  static void InsertDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> attch_node);
  static void InsertControlDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> first_node,
                                         NotNull<AnfNodePtr> second_node);

 private:
  static NotNull<CNodePtr> ProcessKernelGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &last_node,
                                              const CNodePtr &last_label, const VectorRef &args,
                                              NotNull<std::set<KernelGraphPtr> *> memo);
  static void RecurseCall(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node, const CNodePtr &next_node,
                          NotNull<std::set<KernelGraphPtr> *> memo);
  static void RecurseSwitch(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                            NotNull<std::set<KernelGraphPtr> *> memo);
  static void RecurseSwitchLayer(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                                 NotNull<std::set<KernelGraphPtr> *> memo);

  static std::vector<CNodePtr> GetCNodes(const std::vector<AnfNodePtr> &in);
  static void LinkParentGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &from_graph_call_node,
                              const CNodePtr &last_label, const VectorRef &args);
  static void SetSubGraphInput(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> from_graph_call_node,
                               const VectorRef &args);
  static std::tuple<CNodePtr, KernelGraphPtr, VectorRef> ParsePartial(NotNull<AnfNodePtr> node);
  static void InsertAssignToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> from, NotNull<AnfNodePtr> to);
  static size_t SetChildGraphInput(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> node, size_t input_index);

  static constexpr size_t kCNodePrim = 0;
  static constexpr size_t kCNodeCallArg = 1;
  static constexpr size_t kCNodeSwitchCond = 1;
  static constexpr size_t kCNodeSwitchTrue = 2;
  static constexpr size_t kCNodeSwitchFalse = 3;
  static constexpr size_t kCNodeSwitchLength = 4;
  static constexpr size_t kCNodePartialLength = 2;
  static constexpr size_t kCNodePartialFunc = 1;
  static constexpr size_t kCNodeSwitchLayerCond = 1;
  static constexpr size_t kCNodeSwitchLayerBranch = 2;
  static constexpr size_t kCNodeSwitchLayerLength = 3;
};

}  // namespace session
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_SESSION_ASCEND_CONTROL_PARSER_H
