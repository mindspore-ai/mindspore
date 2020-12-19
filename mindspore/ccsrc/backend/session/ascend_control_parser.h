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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_CONTROL_PARSER_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_CONTROL_PARSER_H

#include <set>
#include <map>
#include <vector>
#include <tuple>
#include <utility>
#include <functional>
#include <memory>
#include <string>
#include "backend/session/kernel_graph.h"
#include "base/base_ref.h"
#include "utils/contract.h"
#include "utils/union_find_set.h"

namespace mindspore {
namespace session {
class AscendControlParser {
 public:
  static void LinkGraph(NotNull<KernelGraphPtr> kg);

  static void InsertDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> attch_node);
  static void InsertControlDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> first_node,
                                         NotNull<AnfNodePtr> second_node);
  static void ExecutorValidate(NotNull<KernelGraphPtr> root_graph);
  static void InsertMultipleAssignToGraph(NotNull<KernelGraphPtr> from_graph, const AnfNodePtr &jump_node,
                                          NotNull<AnfNodePtr> from, NotNull<AnfNodePtr> to);

 private:
  class ReferenceCounter;

  static void EraseParameter(NotNull<KernelGraphPtr> root_graph, const std::set<KernelGraphPtr> &graph_list);
  static void EraseAssign(std::shared_ptr<ReferenceCounter> parameter_count, const std::set<CNodePtr> &all_nodes,
                          const std::map<AnfNodePtr, CNodePtr> &para_to_written_node,
                          NotNull<KernelGraphPtr> root_graph, const std::set<KernelGraphPtr> &graph_list);
  static void EraseLabel(NotNull<KernelGraphPtr> root_graph);
  static void ChildGraphDataAssign(NotNull<KernelGraphPtr> kg,
                                   const NotNull<std::vector<std::pair<AnfNodePtr, AnfNodePtr>> *> link_list,
                                   const NotNull<std::set<KernelGraphPtr> *> memo);
  static NotNull<CNodePtr> GetStartLabel(NotNull<KernelGraphPtr> kg, const CNodePtr &last_node,
                                         const CNodePtr &last_label);
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

  static AnfNodePtr InsertAssignToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> from, NotNull<AnfNodePtr> to);
  static std::vector<std::pair<KernelGraphPtr, std::vector<AnfNodePtr>>> ParseCallSwitchNode(
    NotNull<CNodePtr> call_node);
  static std::tuple<KernelGraphPtr, std::vector<AnfNodePtr>> ParsePartial(NotNull<AnfNodePtr> node);
  static void AttachChildGraphToReturnNode(NotNull<KernelGraphPtr> graph,
                                           const NotNull<std::set<KernelGraphPtr> *> memo);
  // root graph order
  static bool CheckLabelIndex(uint32_t index, uint32_t label_index, const CNodePtr &cnode,
                              KernelGraphPtr *cur_child_graph);
  static std::vector<CNodePtr> RecurseGraph(NotNull<KernelGraphPtr> graph,
                                            const NotNull<std::set<KernelGraphPtr> *> memo);
  static void AttachOriginalInputsToGraph(NotNull<KernelGraphPtr> graph, const std::vector<AnfNodePtr> orig_inputs);
};
class AscendControlParser::ReferenceCounter {
 public:
  explicit ReferenceCounter(std::function<bool(int64_t, int64_t)> func) : predicate_(func), count_() {}
  ~ReferenceCounter() = default;
  void AddReadCount(const AnfNodePtr &key, int64_t num);
  void AddWriteCount(const AnfNodePtr &key, int64_t num);
  void EraseElem(const AnfNodePtr &key);
  bool HasValidElem() const;
  std::tuple<AnfNodePtr, int64_t, int64_t> GetOneValidElem() const;

 private:
  std::function<bool(int64_t, int64_t)> predicate_;
  std::map<AnfNodePtr, std::pair<int64_t, int64_t>> count_;
};
}  // namespace session
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_CONTROL_PARSER_H
