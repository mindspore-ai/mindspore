/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/isolate_inplace_func_replace.h"

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "primitive/other_ops.h"
#include "mindspore/core/include/utils/trace_base.h"
#include "frontend/optimizer/irpass/view_inplace_utils.h"
#include "frontend/optimizer/irpass/inplace_input_replace.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
constexpr auto kIsIsolateFuncCallNode = "isolated_call_node";
constexpr auto kIsIsolateInplaceFuncCallNode = "isolated_inplace_call_node";
const char kFuncGraphIsolatedInplaceFunc[] = "isolated_inplace_func";
const char kFuncGraphHasInplaceOp[] = "has_inplace_op";
const char kFuncGraphOnlyIsolatedCalled[] = "only_isolated_called";

class IsolatedInplaceFuncGraphProcesser {
 public:
  IsolatedInplaceFuncGraphProcesser(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager)
      : func_graph_(func_graph), manager_(manager) {
    node_user_map_ = manager_->node_users();
  }
  virtual ~IsolatedInplaceFuncGraphProcesser() = default;
  bool Process();

 private:
  void MarkIsolatedInplaceFuncGraphs();
  void PreProcessIsolateInplaceFuncNodes();
  bool IsIsolatedFuncCallNode(const AnfNodePtr &node);
  bool IsFuncGraphCalledOnlyByIsolatedNode(const FuncGraphPtr &fg);
  bool UsedOnceOnlyByPrim(const AnfNodePtr &node, const PrimitivePtr &prim);
  void DoIsolateCallNodeReplace();
  void AddIsolatedInplaceFunc(const FuncGraphPtr &fg);

  FuncGraphPtr func_graph_{nullptr};
  FuncGraphManagerPtr manager_{nullptr};
  NodeUsersMap node_user_map_;
  std::vector<CNodePtr> isolated_func_call_nodes_;
  FuncGraphVector isolate_func_graphs_;
};

bool MarkInplaceOpFlag(const FuncGraphPtr &func_graph) {
  // Mark inplace op flag for func graph and its sub func_graph
  // Init func_graph with no inplace op
  MS_EXCEPTION_IF_NULL(func_graph);
  func_graph->set_attr(kFuncGraphHasInplaceOp, MakeValue(false));
  const auto &sub_fgs = func_graph->func_graphs_used_total();
  std::for_each(sub_fgs.begin(), sub_fgs.end(),
                [](const FuncGraphPtr &sub_fg) { sub_fg->set_attr(kFuncGraphHasInplaceOp, MakeValue(false)); });

  for (const auto &node : TopoSort(func_graph->return_node(), SuccDeeperSimple)) {
    auto prim = GetCNodePrimitive(node);
    if (prim != nullptr && prim->inplace_prim()) {
      // Mark inplace op flag as true for node's func graph
      auto cur_fg = node->func_graph();
      MS_EXCEPTION_IF_NULL(cur_fg);
      cur_fg->set_attr(kFuncGraphHasInplaceOp, MakeValue(true));
    }
  }
  auto fg_exist_inplace_op = [](const FuncGraphPtr &fg) {
    const auto &cur_sub_fgs = fg->func_graphs_used_total();
    bool exist_inplace_op = std::any_of(cur_sub_fgs.begin(), cur_sub_fgs.end(), [](const FuncGraphPtr &sub_fg) {
      return GetValue<bool>(sub_fg->get_attr(kFuncGraphHasInplaceOp));
    });
    if (exist_inplace_op) {
      fg->set_attr(kFuncGraphHasInplaceOp, MakeValue(true));
    }
  };
  std::for_each(sub_fgs.begin(), sub_fgs.end(), fg_exist_inplace_op);
  fg_exist_inplace_op(func_graph);
  return GetValue<bool>(func_graph->get_attr(kFuncGraphHasInplaceOp));
}

void ModifyFuncGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &params = func_graph->parameters();
  auto original_output = func_graph->output();
  // Replace original_output to MakeTuple(args, original_output)
  AnfNodePtrList output_inputs{NewValueNode(prim::kPrimMakeTuple)};
  output_inputs.insert(output_inputs.end(), params.begin(), params.end());
  (void)output_inputs.emplace_back(original_output);
  auto real_output = func_graph->NewCNodeInOrder(output_inputs);
  func_graph->set_output(real_output);
}

void DoIsolateCallNodeReplaceInner(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  // %1 = Call(func, %0)
  // %2 = Call(func, %0)
  // %3 = Op(%0)
  // ==>
  // %1 = Call(func, %0)
  // %2 = TupleGetItem(%1, 0)
  // %3 = Call(func, %2)
  // %4 = TupleGetItem(%3, 0)
  // %5 = Op(%4)
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(manager);
  const auto &nodes = TopoSort(func_graph->get_return());
  std::unordered_map<AnfNodePtr, AnfNodePtr> node_record_map;
  for (auto &node : nodes) {
    if (!IsCNode(node) || node->func_graph() != func_graph) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    // DoReplace for cnode which use isolated call node's args
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto &cnode_input = cnode->input(i);
      auto it = node_record_map.find(cnode_input);
      if (it == node_record_map.end()) {
        continue;
      }
      auto replaced_node = it->second;
      it = node_record_map.find(replaced_node);
      while (it != node_record_map.end()) {
        replaced_node = it->second;
        it = node_record_map.find(replaced_node);
      }
      MS_LOG(INFO) << "Replace cnode : " << cnode->DebugString() << " input from: " << cnode_input->DebugString()
                   << " to: " << replaced_node->DebugString() << " for isolate inplace func replacement.";
      manager->SetEdge(cnode, i, replaced_node);
    }
    // Record isolated call node's arg as tuplegetitem
    if (!cnode->HasAttr(kIsIsolateInplaceFuncCallNode)) {
      continue;
    }
    // %1(old_output) = {isolate_func_unoin, arg1, arg2, ...}
    // ===>
    // %1(new_tuple_output) = {isolate_func_unoin, arg1, arg2, ...}
    // new_arg1 = {TupleGetItem, %1, 0}
    // ...
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto &cnode_input = cnode->input(i);
      if (HasAbstractMonad(cnode_input)) {
        continue;
      }
      auto idx = MakeValue(SizeToLong(i - 1));
      auto new_arg_cnode =
        func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), cnode, NewValueNode(idx)});
      node_record_map[cnode_input] = new_arg_cnode;
    }
  }
}

bool IsolatedInplaceFuncGraphProcesser::IsIsolatedFuncCallNode(const AnfNodePtr &call_node) {
  MS_EXCEPTION_IF_NULL(call_node);
  if (!IsCNode(call_node)) {
    return false;
  }
  auto call_cnode = call_node->cast<CNodePtr>();
  if (call_cnode->HasAttr(kIsIsolateFuncCallNode)) {
    return true;
  }
  // 1. Only used by updatestate
  // %2 = UpdateState(u, call_node)
  if (UsedOnceOnlyByPrim(call_node, prim::kPrimUpdateState)) {
    call_cnode->AddAttr(kIsIsolateFuncCallNode, MakeValue(true));
    return true;
  }
  // 2. Only used by return node, and this func_graph is called by an isolated node
  // fg1: Return(call_node)
  // %1 = (fg1, args)
  // %2 = UpdateState(u, %1)
  auto cur_fg = call_node->func_graph();
  MS_EXCEPTION_IF_NULL(cur_fg);
  if (UsedOnceOnlyByPrim(call_node, prim::kPrimReturn)) {
    if (cur_fg == func_graph_) {
      return false;
    }
    bool only_called_by_isolated_node = IsFuncGraphCalledOnlyByIsolatedNode(cur_fg);
    if (only_called_by_isolated_node) {
      call_cnode->AddAttr(kIsIsolateFuncCallNode, MakeValue(true));
    }
    return only_called_by_isolated_node;
  }
  // 3. For Isolated switch func call node
  // %1 = {switch, func1, func2}
  // %2 = %1(args)  <== Real call node, check IsIsolatedFuncCallNode
  if (IsPrimitiveCNode(call_node, prim::kPrimSwitch)) {
    auto it = node_user_map_.find(call_node);
    if (it == node_user_map_.end()) {
      return false;
    }
    for (auto &user_info : it->second) {
      auto switch_call_cnode = user_info.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(switch_call_cnode);
      if (!IsIsolatedFuncCallNode(switch_call_cnode)) {
        return false;
      }
      switch_call_cnode->AddAttr(kIsIsolateFuncCallNode, MakeValue(true));
    }
    return true;
  }
  return false;
}

bool IsolatedInplaceFuncGraphProcesser::IsFuncGraphCalledOnlyByIsolatedNode(const FuncGraphPtr &fg) {
  if (fg->has_flag(kFuncGraphIsolatedInplaceFunc)) {
    return true;
  }
  if (fg->has_attr(kFuncGraphOnlyIsolatedCalled)) {
    return GetValue<bool>(fg->get_attr(kFuncGraphOnlyIsolatedCalled));
  }

  MS_EXCEPTION_IF_NULL(fg);
  const auto &cnode_users_map = fg->func_graph_cnodes_index();
  std::vector<CNodePtr> used_cnodes;
  std::for_each(cnode_users_map.begin(), cnode_users_map.end(),
                [&used_cnodes, this](const std::pair<const CNodeIndexPairPtr, int64_t> &item) {
                  auto node = item.first->first;
                  MS_EXCEPTION_IF_NULL(node);
                  auto cnode = node->cast<CNodePtr>();
                  MS_EXCEPTION_IF_NULL(cnode);
                  // Remove unused nodes, such as Partial eliminated cnodes
                  if (node_user_map_.find(cnode) != node_user_map_.end()) {
                    (void)used_cnodes.emplace_back(cnode);
                  }
                });
  MS_LOG(DEBUG) << "Start checking if func graph is only called by isolated nodes, fg: " << fg->ToString()
                << " , used node size: " << used_cnodes.size();
  if (used_cnodes.empty()) {
    fg->set_attr(kFuncGraphOnlyIsolatedCalled, MakeValue(false));
    return false;
  }
  // Init current fg's flag to avoid endless loop search
  fg->set_attr(kFuncGraphOnlyIsolatedCalled, MakeValue(fg == func_graph_));
  auto first_call_node = used_cnodes.front();
  bool only_called_by_isolated_node = IsIsolatedFuncCallNode(first_call_node);
  for (size_t i = 1; i < used_cnodes.size(); ++i) {
    auto call_node = used_cnodes[i];
    bool cur_used_by_isolated_node = IsIsolatedFuncCallNode(call_node);
    if (only_called_by_isolated_node != cur_used_by_isolated_node &&
        GetValue<bool>(fg->get_attr(kFuncGraphHasInplaceOp))) {
      MS_LOG(WARNING)
        << "For function containing inplace operators, since the caller nodes contain isolated nodes and "
           "non-isolated nodes, may be a loop function, computing grad for such functions may affect accuracy, fg: "
        << fg->ToString() << ". Please modify the code to prevent isolated function calls, "
        << "code location: " << trace::GetDebugInfoStr(call_node->debug_info());
    }
    only_called_by_isolated_node &= cur_used_by_isolated_node;
  }
  if (only_called_by_isolated_node) {
    MS_LOG(INFO) << "Mark fg: " << fg->ToString() << "as only isolated called func graph.";
  }
  MS_LOG(DEBUG) << "Finish checking if func graph is only called by isolated nodes, fg: " << fg->ToString()
                << " , flag: " << only_called_by_isolated_node;
  fg->set_attr(kFuncGraphOnlyIsolatedCalled, MakeValue(only_called_by_isolated_node));
  return only_called_by_isolated_node;
}

void IsolatedInplaceFuncGraphProcesser::MarkIsolatedInplaceFuncGraphs() {
  // Mark func graph which is both has inplace op and only called by isolated nodes
  for (auto &fg : func_graph_->func_graphs_used_total()) {
    auto inplace_op_flag = GetValue<bool>(fg->get_attr(kFuncGraphHasInplaceOp));
    if (inplace_op_flag && IsFuncGraphCalledOnlyByIsolatedNode(fg)) {
      AddIsolatedInplaceFunc(fg);
    }
  }
}

void IsolatedInplaceFuncGraphProcesser::PreProcessIsolateInplaceFuncNodes() {
  for (auto &node : TopoSort(func_graph_->get_return(), SuccDeeperSimple)) {
    if (!IsCNode(node) || node->func_graph() == nullptr) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!cnode->HasAttr(kIsIsolateFuncCallNode)) {
      continue;
    }
    const auto &inputs = cnode->inputs();
    auto first_input_node = inputs[kIndex0];
    // Support isolate func nodes called once now, loop or graph reuse scene need handle later
    auto fg = GetValueNode<FuncGraphPtr>(first_input_node);
    if (fg != nullptr && fg->has_flag(kFuncGraphIsolatedInplaceFunc)) {
      (void)isolated_func_call_nodes_.emplace_back(cnode);
      MS_LOG(INFO) << "Found isolated inplace func call node: " << cnode->DebugString()
                   << ", modify func graph: " << fg->ToString();
    } else if (IsPrimitiveCNode(first_input_node, prim::kPrimSwitch)) {
      // Find %2 as isolate inplace func call in switch situation
      // %1 = Switch(cond, func1, isolate_func2)
      // %2 = %1(args)  ==> isolated_call_node
      auto switch_cnode = first_input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(switch_cnode);
      const auto &switch_inputs = switch_cnode->inputs();
      auto fg1_node = switch_inputs[kIndex2];
      auto fg2_node = switch_inputs[kIndex3];
      auto fg1 = GetValueNode<FuncGraphPtr>(fg1_node);
      auto fg2 = GetValueNode<FuncGraphPtr>(fg2_node);
      bool isolated_inplace_func =
        fg1->has_flag(kFuncGraphIsolatedInplaceFunc) || fg2->has_flag(kFuncGraphIsolatedInplaceFunc);
      if (isolated_inplace_func) {
        if (!IsFuncGraphCalledOnlyByIsolatedNode(fg1) || !IsFuncGraphCalledOnlyByIsolatedNode(fg2)) {
          MS_LOG(EXCEPTION) << "For isolated inplace func switch call node, both branch should be called "
                            << "only by isolated nodes, but got error func graph reusing, fg1: " << fg1->ToString()
                            << " and fg2: " << fg2->ToString() << ". Please modify the code to prevent isolated "
                            << "function calls, code location: " << trace::GetDebugInfoStr(cnode->debug_info());
        }
        (void)isolated_func_call_nodes_.emplace_back(cnode);
        AddIsolatedInplaceFunc(fg1);
        AddIsolatedInplaceFunc(fg2);
        MS_LOG(INFO) << "Found isolated inplace func switch call node: " << switch_cnode->DebugString()
                     << ", modify func graph: " << fg1->ToString() << " , and: " << fg2->ToString();
      }
    }
  }
}

bool IsolatedInplaceFuncGraphProcesser::UsedOnceOnlyByPrim(const AnfNodePtr &node, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(prim);
  auto it = node_user_map_.find(node);
  if (it == node_user_map_.end()) {
    return false;
  }
  constexpr size_t node_use_times = 1;
  const auto &user_set = it->second;
  return user_set.size() == node_use_times && IsPrimitiveCNode(user_set.front().first, prim);
}

void IsolatedInplaceFuncGraphProcesser::DoIsolateCallNodeReplace() {
  DoIsolateCallNodeReplaceInner(func_graph_, manager_);
  for (auto &fg : func_graph_->func_graphs_used_total()) {
    DoIsolateCallNodeReplaceInner(fg, manager_);
  }
}

void IsolatedInplaceFuncGraphProcesser::AddIsolatedInplaceFunc(const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  if (!fg->has_flag(kFuncGraphIsolatedInplaceFunc)) {
    (void)isolate_func_graphs_.emplace_back(fg);
    fg->set_flag(kFuncGraphIsolatedInplaceFunc, true);
    MS_LOG(INFO) << "Mark fg : " << fg->ToString() << " as isolated inplace func graph";
  }
}

bool IsolatedInplaceFuncGraphProcesser::Process() {
  // Set flag for func_graph which only called by isolated node and has inplace_op
  MarkIsolatedInplaceFuncGraphs();
  if (isolate_func_graphs_.empty()) {
    return false;
  }
  PreProcessIsolateInplaceFuncNodes();
  // Change isolated inplace func call's output as tuple(args)
  // Replace following original args as corresponding tuple getitem
  for (auto &isolated_call_node : isolated_func_call_nodes_) {
    isolated_call_node->AddAttr(kIsIsolateInplaceFuncCallNode, MakeValue(True));
  }
  for (auto &sub_fg : isolate_func_graphs_) {
    ModifyFuncGraph(sub_fg);
  }
  DoIsolateCallNodeReplace();
  return true;
}
}  // namespace

bool IsolateInplaceFuncReplace(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!MarkInplaceOpFlag(func_graph)) {
    return false;
  }
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto func_processor = std::make_shared<IsolatedInplaceFuncGraphProcesser>(func_graph, manager);
  auto need_process = func_processor->Process();
  if (need_process) {
    (void)DoInplaceInputReplace(func_graph, optimizer);
  }
  return need_process;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
