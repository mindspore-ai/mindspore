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

#include "backend/session/ascend_control_parser.h"
#include <utility>
#include <memory>
#include <algorithm>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/union_find_set.h"
#include "runtime/device/ascend/ascend_label_assign.h"
#include "utils/ms_context.h"
#include "debug/anf_ir_dump.h"

static constexpr size_t kCNodePrim = 0;
static constexpr size_t kCNodeCallArg = 1;
static constexpr size_t kCNodeSwitchCond = 1;
static constexpr size_t kCNodeSwitchTrue = 2;
static constexpr size_t kCNodeSwitchFalse = 3;
static constexpr size_t kCNodeSwitchLength = 4;
static constexpr size_t kCNodePartialLength = 2;
static constexpr size_t kCNodePartialFunc = 1;
static constexpr size_t kCNodeSwitchLayerBranch = 2;
static constexpr size_t kCNodeSwitchLayerLength = 3;
static constexpr size_t kCNodeAssignTarget = 1;
static constexpr size_t kCNodeAssignSource = 2;
static constexpr size_t kCNodeAssignDestination = 1;

namespace mindspore {
namespace session {
static void RecursiveReplaceNode(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> main_parameter,
                                 const std::set<AnfNodePtr> &parameter_reuse_set,
                                 const NotNull<std::set<KernelGraphPtr> *> memo) {
  if (parameter_reuse_set.empty()) {
    MS_LOG(EXCEPTION) << "Parameter_reuse_set is empty.";
  }
  if (memo->find(kg.get()) != memo->end()) {
    return;
  }
  memo->insert(kg.get());

  for (auto &para : parameter_reuse_set) {
    if (para == main_parameter.get()) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(para);
    MS_LOG(INFO) << "In " << kg->ToString() << " replace " << para->DebugString() << " of graph "
                 << AnfAlgo::GetGraphId(para.get()) << " to " << main_parameter->DebugString() << " of graph "
                 << AnfAlgo::GetGraphId(main_parameter.get().get());
    kg->ReplaceNode(NOT_NULL(para), main_parameter);
  }

  for (auto &child : kg->child_graph_order()) {
    RecursiveReplaceNode(NOT_NULL(child.lock()), main_parameter, parameter_reuse_set, memo);
  }
}

static AnfNodePtr GetMainParameter(NotNull<KernelGraphPtr> root_kg, const AnfNodePtr &key,
                                   const std::set<AnfNodePtr> &parameter_reuse_set) {
  AnfNodePtr main_parameter = key;
  std::set<AnfNodePtr> root_inputs_set;
  const auto &root_inputs_vector = root_kg->inputs();
  root_inputs_set.insert(root_inputs_vector.begin(), root_inputs_vector.end());
  for (auto &node : parameter_reuse_set) {
    if (root_inputs_set.find(node) != root_inputs_set.end()) {
      main_parameter = node;
      break;
    }
  }
  return main_parameter;
}

static void ReuseParameter(NotNull<KernelGraphPtr> root_kg,
                           const std::vector<std::pair<AnfNodePtr, AnfNodePtr>> &link_list) {
  // make union find set
  UnionFindSet<AnfNodePtr> union_find_set;
  for (auto &[param, arg] : link_list) {
    union_find_set.Add(param);
    union_find_set.Add(arg);
  }
  for (auto &[param, arg] : link_list) {
    union_find_set.Union(param, arg);
  }
  auto parameter_reuse_sets = union_find_set.GetSets();

  for (auto &[key, parameter_reuse_set] : parameter_reuse_sets) {
    if (parameter_reuse_set.size() <= 1) {
      continue;
    }
    auto main_parameter = GetMainParameter(root_kg, key, parameter_reuse_set);
    std::set<KernelGraphPtr> memo;
    RecursiveReplaceNode(root_kg, NOT_NULL(main_parameter), parameter_reuse_set, NOT_NULL(&memo));
  }
}

static CNodePtr GetNextRealKernel(const std::vector<CNodePtr> &list, size_t start) {
  for (size_t i = start; i < list.size() - 1; ++i) {
    if (AnfAlgo::IsRealKernel(list[i])) {
      return list[i];
    }
  }
  return nullptr;
}

static void UpdateLabelIdToLabelSetMap(const std::vector<CNodePtr> &exec_order,
                                       const NotNull<std::map<uint32_t, CNodePtr> *> label_id_to_label_set) {
  for (auto &node : exec_order) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimLabelSet)) {
      continue;
    }
    if (!AnfAlgo::HasNodeAttr(kAttrLabelIndex, node)) {
      MS_LOG(EXCEPTION) << node->DebugString() << " has no attr kAttrLabelIndex";
    }
    uint32_t label_id = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
    if (auto iter = label_id_to_label_set->find(label_id); iter != label_id_to_label_set->end()) {
      MS_LOG(EXCEPTION) << "There are more than one node has same label id " << label_id
                        << ", node: " << iter->second->DebugString() << " and " << node->DebugString();
    }
    (*label_id_to_label_set)[label_id] = node;
  }
}

static std::vector<CNodePtr> GetTargetLabelSetNodes(NotNull<CNodePtr> jump_node,
                                                    const std::map<uint32_t, CNodePtr> &label_id_to_label_set) {
  std::vector<uint32_t> target_label_list;
  std::vector<CNodePtr> target_labelset_nodes;
  if (IsPrimitiveCNode(jump_node.get(), prim::kPrimLabelGoto)) {
    if (!AnfAlgo::HasNodeAttr(kAttrLabelIndex, jump_node)) {
      MS_LOG(EXCEPTION) << jump_node->DebugString() << " has no attr kAttrLabelIndex";
    }
    uint32_t label_id = AnfAlgo::GetNodeAttr<uint32_t>(jump_node.get(), kAttrLabelIndex);
    target_label_list.push_back(label_id);
  } else if (IsPrimitiveCNode(jump_node.get(), prim::kPrimLabelSwitch)) {
    if (!AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, jump_node)) {
      MS_LOG(EXCEPTION) << jump_node->DebugString() << " has no attr kPrimLabelSwitch";
    }
    target_label_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(jump_node.get(), kAttrLabelSwitchList);
  } else {
    MS_LOG(EXCEPTION) << "Unknown type jump node " << jump_node->DebugString();
  }

  for (auto label_id : target_label_list) {
    auto iter = label_id_to_label_set.find(label_id);
    if (iter == label_id_to_label_set.end()) {
      MS_LOG(EXCEPTION) << "Cannot find LabelSet node has label id " << label_id;
    }
    target_labelset_nodes.push_back(iter->second);
  }
  return target_labelset_nodes;
}

static void EraseNodeFromExecOrder(const AnfNodePtr &node, const NotNull<std::vector<CNodePtr> *> exec_order) {
  MS_EXCEPTION_IF_NULL(node);
  auto exec_iter = std::find(exec_order->begin(), exec_order->end(), node);
  if (exec_iter == exec_order->end()) {
    MS_LOG(EXCEPTION) << "Cannot find " << node->DebugString() << " in exec order.";
  }
  exec_order->erase(exec_iter);
}

void AscendControlParser::AttachChildGraphToReturnNode(NotNull<KernelGraphPtr> graph,
                                                       const NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph.get());
  const std::vector<std::weak_ptr<KernelGraph>> &child_graph_order = graph->child_graph_order();
  if (child_graph_order.empty()) {
    return;
  }

  std::vector<AnfNodePtr> depend_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  for (auto &kg : child_graph_order) {
    std::shared_ptr<KernelGraph> cg = kg.lock();
    MS_EXCEPTION_IF_NULL(cg);
    auto fg = cg->cast<FuncGraphPtr>();
    MS_EXCEPTION_IF_NULL(fg);
    depend_inputs.emplace_back(NewValueNode(fg));
    AttachChildGraphToReturnNode(NOT_NULL(cg), memo);
  }
  auto child_graphs = graph->NewCNode(depend_inputs);
  InsertDependToGraph(graph, NOT_NULL(child_graphs));
}

void AscendControlParser::LinkGraph(NotNull<KernelGraphPtr> kg) {
  std::set<KernelGraphPtr> memo;
  std::vector<std::pair<AnfNodePtr, AnfNodePtr>> link_list;
  // Insert Assign
  ChildGraphDataAssign(kg, NOT_NULL(&link_list), NOT_NULL(&memo));
  memo.clear();
  // Reuse Parameter
  ReuseParameter(kg, link_list);
  // replace call by label goto / label switch
  (void)ProcessKernelGraph(kg, nullptr, nullptr, NOT_NULL(&memo));
  memo.clear();
  // assign label resource
  device::ascend::AscendLabelAssign::GetInstance().AssignLabel(kg);
}

void AscendControlParser::EraseParameter(NotNull<KernelGraphPtr> root_graph,
                                         const std::set<KernelGraphPtr> &graph_list) {
  std::vector<CNodePtr> exec_order = root_graph->execution_order();
  std::set<CNodePtr> search_list(exec_order.begin(), exec_order.end());
  std::set<AnfNodePtr> root_inputs(root_graph->inputs().begin(), root_graph->inputs().end());
  auto ref_map = root_graph->GetRefMap();
  ReferenceCounter parameter_count([](int64_t read, int64_t write) -> bool { return write == 1; });
  std::multimap<AnfNodePtr, std::tuple<size_t, AnfNodePtr, size_t>> ref_multimap;
  std::transform(ref_map.begin(), ref_map.end(), std::inserter(ref_multimap, ref_multimap.end()),
                 [](const std::pair<std::pair<AnfNodePtr, size_t>, std::pair<AnfNodePtr, size_t>> &p)
                   -> std::pair<AnfNodePtr, std::tuple<size_t, AnfNodePtr, size_t>> {
                   return {p.first.first, {p.first.second, p.second.first, p.second.second}};
                 });
  std::set<CNodePtr> all_nodes;
  std::map<AnfNodePtr, CNodePtr> para_to_written_node;
  for (auto &graph : graph_list) {
    auto out = graph->get_return();
    MS_EXCEPTION_IF_NULL(out);
    search_list.insert(out->cast<CNodePtr>());
    auto nodes = TopoSort(out);
    for (auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      auto cnode = node->cast<CNodePtr>();
      if (cnode != nullptr) {
        all_nodes.insert(cnode);
      }
    }
  }
  // parameter->transdata->assign<-5d node, ref parameter would get from transdata input
  auto validate_ref_parameter = [](AnfNodePtr node) -> AnfNodePtr {
    if (node->isa<CNode>() && AnfAlgo::CheckPrimitiveType(node, prim::KPrimTransData)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto first_input = cnode->input(kFirstDataInputIndex);
      MS_EXCEPTION_IF_NULL(first_input);
      return first_input;
    }
    return node;
  };
  // prepare referance count
  for (auto &node : search_list) {
    MS_EXCEPTION_IF_NULL(node);
    // if assign node
    std::set<AnfNodePtr> refed_parameters;
    for (auto [iter, end] = ref_multimap.equal_range(node); iter != end; ++iter) {
      refed_parameters.insert(validate_ref_parameter(std::get<1>(iter->second)));
    }

    for (auto &in : node->inputs()) {
      auto visit_node = AnfAlgo::VisitKernelWithReturnType(in, 0).first;
      visit_node = validate_ref_parameter(visit_node);
      if (!visit_node->isa<Parameter>() || root_inputs.find(visit_node) != root_inputs.end()) {
        continue;
      }
      if (refed_parameters.find(visit_node) != refed_parameters.end()) {
        parameter_count.AddWriteCount(visit_node, 1);
        para_to_written_node[visit_node] = node;
      } else {
        parameter_count.AddReadCount(visit_node, 1);
      }
    }
  }

  EraseAssign(std::make_shared<ReferenceCounter>(parameter_count), all_nodes, para_to_written_node, root_graph,
              graph_list);
}

void AscendControlParser::EraseAssign(std::shared_ptr<ReferenceCounter> parameter_count,
                                      const std::set<CNodePtr> &all_nodes,
                                      const std::map<AnfNodePtr, CNodePtr> &para_to_written_node,
                                      NotNull<KernelGraphPtr> root_graph, const std::set<KernelGraphPtr> &graph_list) {
  std::vector<CNodePtr> exec_order = root_graph->execution_order();
  while (parameter_count->HasValidElem()) {
    auto [para, read, written] = parameter_count->GetOneValidElem();
    MS_LOG(INFO) << para->DebugString() << " was read " << read << " times, written " << written << " times.";
    auto assign_iter = para_to_written_node.find(para);
    if (assign_iter == para_to_written_node.end()) {
      MS_LOG(EXCEPTION) << "Cannot find assign node that write " << para->DebugString();
    }
    auto &assign_node = assign_iter->second;
    MS_EXCEPTION_IF_NULL(assign_node);
    auto source = assign_node->input(kCNodeAssignSource);
    auto destination = assign_node->input(kCNodeAssignDestination);
    // not assign node or assign destination is transdata which for ref parameter(write 2 times) -> continue
    if (!IsPrimitiveCNode(assign_node, prim::kPrimAssign) || IsPrimitiveCNode(destination, prim::KPrimTransData)) {
      parameter_count->EraseElem(para);
      continue;
    }
    MS_LOG(INFO) << "Erase " << assign_node->DebugString(5);
    EraseNodeFromExecOrder(assign_node, NOT_NULL(&exec_order));
    MS_EXCEPTION_IF_NULL(source);
    auto visit_source = AnfAlgo::VisitKernelWithReturnType(source, 0).first;
    parameter_count->AddWriteCount(para, -1);
    parameter_count->AddReadCount(para, -1);
    if (visit_source->isa<Parameter>()) {
      parameter_count->AddReadCount(visit_source, read - 1);
    }

    // replace parameter in node
    for (auto &node : all_nodes) {
      for (size_t i = 0; i < node->size(); ++i) {
        if (node->input(i) == para) {
          MS_LOG_INFO << "Replace " << node->DebugString() << " input " << i << " by " << source->DebugString();
          node->set_input(i, source);
        }
      }
    }

    // replace parameter in graph input
    for (auto &g : graph_list) {
      auto child_graph_inputs = g->MutableInputs();
      std::replace(child_graph_inputs->begin(), child_graph_inputs->end(), para, source);
      MS_LOG_INFO << "Replace parameter " << para->DebugString() << " by " << source->DebugString() << " in graph "
                  << g->graph_id() << " inputs";
    }
  }
  root_graph->set_execution_order(exec_order);
}

void AscendControlParser::EraseLabel(NotNull<KernelGraphPtr> root_graph) {
  std::vector<CNodePtr> exec_order = root_graph->execution_order();
  ReferenceCounter label_count([](int32_t read, int32_t write) -> bool { return read <= 1; });
  std::map<AnfNodePtr, CNodePtr> label_to_written_node;
  std::map<uint32_t, CNodePtr> label_id_to_label_set;
  UpdateLabelIdToLabelSetMap(exec_order, NOT_NULL(&label_id_to_label_set));
  CNodePtr last_node = nullptr;
  for (auto &cur_node : exec_order) {
    MS_EXCEPTION_IF_NULL(cur_node);
    if (AnfAlgo::IsCondControlKernel(cur_node)) {
      std::vector<CNodePtr> target_labelset_nodes = GetTargetLabelSetNodes(NOT_NULL(cur_node), label_id_to_label_set);
      for (auto &label_set : target_labelset_nodes) {
        label_count.AddReadCount(label_set, 1);
        label_to_written_node[label_set] = cur_node;
      }
    } else if (IsPrimitiveCNode(cur_node, prim::kPrimLabelSet)) {
      label_count.AddWriteCount(cur_node, 1);
      if (last_node != nullptr && !AnfAlgo::IsCondControlKernel(last_node)) {
        label_count.AddReadCount(cur_node, 1);
        label_to_written_node[cur_node] = last_node;
      }
    }
    last_node = cur_node;
  }

  while (label_count.HasValidElem()) {
    auto [label_set, read, written] = label_count.GetOneValidElem();
    MS_LOG(INFO) << label_set->DebugString() << " was read " << read << " times, written " << written << " times.";
    auto iter = label_to_written_node.find(label_set);
    if (read > 0 && iter == label_to_written_node.end()) {
      MS_LOG(EXCEPTION) << "Cannot find node jump to " << label_set->DebugString();
    }
    CNodePtr jump_node = read > 0 ? iter->second : nullptr;
    if (jump_node == nullptr || IsPrimitiveCNode(jump_node, prim::kPrimLabelGoto)) {
      MS_LOG(INFO) << "Erase node " << label_set->DebugString();
      EraseNodeFromExecOrder(label_set, NOT_NULL(&exec_order));
    }
    if (jump_node != nullptr && IsPrimitiveCNode(jump_node, prim::kPrimLabelGoto)) {
      MS_LOG(INFO) << "Erase node " << jump_node->DebugString();
      EraseNodeFromExecOrder(jump_node, NOT_NULL(&exec_order));
    }
    label_count.EraseElem(label_set);
  }

  root_graph->set_execution_order(exec_order);
}

void AscendControlParser::ExecutorValidate(NotNull<KernelGraphPtr> root_graph) {
  std::set<KernelGraphPtr> memo;
  (void)RecurseGraph(root_graph, NOT_NULL(&memo));
  EraseParameter(root_graph, memo);
  EraseLabel(root_graph);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    std::string file_name = "after_erase_label_and_parameter.ir";
    DumpIR(file_name, root_graph.get());
  }
}

std::vector<std::pair<KernelGraphPtr, std::vector<AnfNodePtr>>> AscendControlParser::ParseCallSwitchNode(
  NotNull<CNodePtr> cnode) {
  std::vector<std::pair<KernelGraphPtr, std::vector<AnfNodePtr>>> ret;

  if (IsPrimitiveCNode(cnode.get(), prim::kPrimCall)) {
    if (cnode->size() <= kCNodeCallArg) {
      MS_LOG(EXCEPTION) << "Call node " << cnode->DebugString() << " has invalid inputs size " << cnode->size();
    }
    auto call_arg = cnode->input(kCNodeCallArg);
    MS_EXCEPTION_IF_NULL(call_arg);
    ret.emplace_back(GetValueNode<KernelGraphPtr>(call_arg),
                     std::vector<AnfNodePtr>(cnode->inputs().begin() + kCNodeCallArg + 1, cnode->inputs().end()));
  } else if (IsPrimitiveCNode(cnode.get(), prim::kPrimSwitch)) {
    const std::vector<AnfNodePtr> &switch_inputs = cnode->inputs();
    if (switch_inputs.size() < kCNodeSwitchLength) {
      MS_LOG(EXCEPTION) << "Switch node " << cnode->DebugString() << " has invalid inputs size "
                        << switch_inputs.size();
    }
    for (auto iter = switch_inputs.begin() + kCNodeSwitchCond + 1; iter != switch_inputs.end(); ++iter) {
      const auto &[target_graph, args] = ParsePartial(NOT_NULL(*iter));
      ret.emplace_back(target_graph, args);
    }
  } else if (IsPrimitiveCNode(cnode.get(), prim::kPrimSwitchLayer)) {
    const std::vector<AnfNodePtr> &switch_layer_inputs = cnode->inputs();
    if (switch_layer_inputs.size() <= kCNodeSwitchLayerBranch) {
      MS_LOG(EXCEPTION) << "Switch layer node " << cnode->DebugString() << " has invalid inputs size "
                        << switch_layer_inputs.size();
    }
    for (auto iter = switch_layer_inputs.begin() + kCNodeSwitchLayerBranch; iter != switch_layer_inputs.end(); ++iter) {
      const auto &[target_graph, args] = ParsePartial(NOT_NULL(*iter));
      ret.emplace_back(target_graph, args);
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported call node: " << cnode->DebugString(5);
  }
  return ret;
}

void AscendControlParser::ChildGraphDataAssign(
  NotNull<KernelGraphPtr> kg, const NotNull<std::vector<std::pair<AnfNodePtr, AnfNodePtr>> *> link_list,
  const NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(kg) != memo->end()) {
    return;
  }
  memo->insert(kg.get());

  MS_LOG(INFO) << "Start link data for " << kg->ToString();
  const std::vector<CNodePtr> &nodes = kg->execution_order();

  for (auto &node : nodes) {
    if (!(IsPrimitiveCNode(node, prim::kPrimCall) || IsPrimitiveCNode(node, prim::kPrimSwitch) ||
          IsPrimitiveCNode(node, prim::kPrimSwitchLayer))) {
      continue;
    }

    auto child_graph_list = ParseCallSwitchNode(NOT_NULL(node));
    for (auto &[child_graph, args] : child_graph_list) {
      MS_EXCEPTION_IF_NULL(child_graph);
      const std::vector<AnfNodePtr> &params = child_graph->inputs();
      if (args.size() != params.size()) {
        MS_LOG(EXCEPTION) << child_graph->ToString() << " needs " << params.size() << " inputs but call node "
                          << node->DebugString(5) << " gives " << args.size();
      }
      for (size_t i = 0; i < args.size(); ++i) {
        InsertMultipleAssignToGraph(kg, node, NOT_NULL(args[i]), NOT_NULL(params[i]));
      }
    }
  }
  kg->SetExecOrderByDefault();
  for (auto &child_graph : kg->child_graph_order()) {
    ChildGraphDataAssign(NOT_NULL(child_graph.lock()), link_list, memo);
  }
}

NotNull<CNodePtr> AscendControlParser::GetStartLabel(NotNull<KernelGraphPtr> kg, const CNodePtr &last_node,
                                                     const CNodePtr &last_label) {
  CNodePtr start_label;
  if (last_node != nullptr && last_label != nullptr) {
    start_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
    MS_LOG(INFO) << "Insert start label " << start_label->DebugString() << " to " << kg->ToString();
    kg->set_start_label(start_label);
  } else {
    // no goto node will jump to start label of root graph, so return a fake label
    start_label = std::make_shared<CNode>(std::vector<AnfNodePtr>(), FuncGraphPtr(nullptr));
  }
  return NOT_NULL(start_label);
}

NotNull<CNodePtr> AscendControlParser::ProcessKernelGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &last_node,
                                                          const CNodePtr &last_label,
                                                          const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "Start process KernelGraph " << kg->ToString();

  // 1. recursive condition
  if (memo->find(kg) != memo->end()) {
    MS_LOG(INFO) << "KernelGraph has beed processed: " << kg->ToString();
    return NOT_NULL(kg->get_start_label());
  }
  memo->insert(kg.get());

  // 2. args replace placeholder
  LinkParentGraph(kg, last_node, last_label);

  // 3. topological sort
  kg->SetExecOrderByDefault();
  const std::vector<CNodePtr> &nodes = kg->execution_order();
  // 4. insert first_label
  CNodePtr start_label = GetStartLabel(kg, last_node, last_label);

  // 5. traverse
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto &cnode = nodes[i];
    MS_EXCEPTION_IF_NULL(cnode);
    if (!(AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall) ||
          AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch) ||
          AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer))) {
      continue;
    }

    if (IsPrimitiveCNode(cnode, prim::kPrimCall)) {
      RecurseCall(kg, NOT_NULL(cnode), GetNextRealKernel(nodes, i + 1), memo);
    } else if (IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
      RecurseSwitch(kg, NOT_NULL(cnode), GetNextRealKernel(nodes, i + 1), memo);
    } else if (IsPrimitiveCNode(cnode, prim::kPrimSwitchLayer)) {
      RecurseSwitchLayer(kg, NOT_NULL(cnode), GetNextRealKernel(nodes, i + 1), memo);
    } else {
      MS_LOG(EXCEPTION) << "Unexpected node: " << cnode->DebugString();
    }
  }
  kg->SetExecOrderByDefault();
  MS_LOG(INFO) << "End KernelGraph process: " << kg->ToString();
  return NOT_NULL(start_label);
}

void AscendControlParser::InsertDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> attch_node) {
  auto return_node = kg->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                    return_node->input(kFirstDataInputIndex), attch_node.get()};
  auto depend_node = kg->NewCNode(inputs);
  return_node->set_input(kFirstDataInputIndex, depend_node);
}

void AscendControlParser::InsertControlDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> first_node,
                                                     NotNull<AnfNodePtr> second_node) {
  MS_LOG(INFO) << "Insert control depend at the end of graph, the first node is " << first_node->DebugString()
               << ", the second node is " << second_node->DebugString();
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimControlDepend->name())),
                                    first_node, second_node};
  auto control_depend = kg->NewCNode(inputs);
  InsertDependToGraph(kg, NOT_NULL(control_depend));
}

void AscendControlParser::LinkParentGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &from_graph_call_node,
                                          const CNodePtr &last_label) {
  // if not entry graph, replace return with label_goto
  if (from_graph_call_node != nullptr && last_label != nullptr) {
    auto label_goto =
      kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelGotoOpName)), last_label});
    MS_EXCEPTION_IF_NULL(label_goto);
    MS_LOG(INFO) << "Insert end goto " << label_goto->DebugString() << " to " << kg->ToString();
    kg->set_end_goto(label_goto);
  }
}

void AscendControlParser::AttachOriginalInputsToGraph(NotNull<KernelGraphPtr> graph,
                                                      const std::vector<AnfNodePtr> orig_inputs) {
  std::vector<AnfNodePtr> make_tuple_inputs = {
    mindspore::NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  std::copy(orig_inputs.begin(), orig_inputs.end(), std::back_inserter(make_tuple_inputs));
  auto make_tuple = graph->NewCNode(make_tuple_inputs);

  InsertDependToGraph(graph, NOT_NULL(make_tuple));
}

void AscendControlParser::RecurseCall(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node, const CNodePtr &next_node,
                                      const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "Process call func " << cur_node->DebugString();

  // 1 get kernel graph
  std::vector<AnfNodePtr> origin_inputs = cur_node->inputs();
  if (kCNodeCallArg >= origin_inputs.size()) {
    MS_LOG(EXCEPTION) << "Index out of range,size:" << origin_inputs.size();
  }
  std::vector<AnfNodePtr> new_inputs = {std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelGotoOpName))};
  if (!IsValueNode<KernelGraph>(origin_inputs[kCNodeCallArg])) {
    MS_LOG(WARNING) << "Node " << cur_node->DebugString(10) << " index " << kCNodeCallArg << " is not a ValueNode";
    return;
  }
  // 2 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  MS_LOG(INFO) << "Insert back label " << back_label->DebugString() << " to " << kg->ToString() << " call node "
               << cur_node->DebugString();
  // 3 add depend relationship
  InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
  if (next_node != nullptr && next_node != kg->get_return()) {
    InsertControlDependToGraph(kg, NOT_NULL(back_label), NOT_NULL(next_node));
  }
  auto call_kg = GetValueNode<KernelGraphPtr>(origin_inputs[kCNodeCallArg]);
  // 4 modify call op to goto op
  cur_node->set_input(kCNodePrim, new_inputs[kCNodePrim]);
  // 5 recurse sub graph
  CNodePtr sub_label = ProcessKernelGraph(NOT_NULL(call_kg), cur_node, back_label, memo);
  new_inputs.push_back(sub_label);
  cur_node->set_inputs(new_inputs);
  cur_node->set_abstract(nullptr);
  AnfAlgo::SetNodeAttr(kAttrChildGraph, MakeValue<std::vector<KernelGraphPtr>>({call_kg}), cur_node.get());
  kg->RemoveNodeFromGraph(origin_inputs[kCNodeCallArg]);
  origin_inputs.assign(origin_inputs.begin() + kCNodeCallArg + 1, origin_inputs.end());
  AttachOriginalInputsToGraph(kg, origin_inputs);
  MS_LOG(INFO) << "Succeed processing call func " << cur_node->DebugString();
}

void AscendControlParser::RecurseSwitch(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                                        const CNodePtr &next_node, const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "Process switch node " << cur_node->DebugString();

  if (cur_node->size() < kCNodeSwitchLength) {
    MS_LOG(EXCEPTION) << "Inputs of apply node must more than " << kCNodeSwitchLength;
  }
  // 1 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  MS_EXCEPTION_IF_NULL(back_label);
  MS_LOG(INFO) << "Insert back label " << back_label->DebugString() << " to " << kg->ToString() << " switch node "
               << cur_node->DebugString();
  // 2 add depend relationship
  InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
  if (next_node != nullptr && next_node != kg->get_return()) {
    InsertControlDependToGraph(kg, NOT_NULL(back_label), NOT_NULL(next_node));
  }
  // 3 recurse sub graph
  const std::vector<AnfNodePtr> &origin_switch_inputs = cur_node->inputs();
  if (kCNodeSwitchCond >= origin_switch_inputs.size()) {
    MS_LOG(EXCEPTION) << "The size of origin_switch_inputs is not more than " << kCNodeSwitchCond;
  }
  std::vector<AnfNodePtr> new_switch_inputs = {
    std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
    origin_switch_inputs[kCNodeSwitchCond]};
  std::vector<KernelGraphPtr> child_graphs;
  for (size_t i = kCNodeSwitchCond + 1; i < kCNodeSwitchLength; ++i) {
    // 3.1 branch kernel graph and args
    KernelGraphPtr branch_fg;
    std::vector<AnfNodePtr> origin_inputs;
    std::tie(branch_fg, origin_inputs) = ParsePartial(NOT_NULL(origin_switch_inputs[i]));
    child_graphs.push_back(branch_fg);
    // 3.2 recurse sub graph
    CNodePtr branch_label = ProcessKernelGraph(NOT_NULL(branch_fg), cur_node, back_label, memo);
    new_switch_inputs.push_back(branch_label);
    AttachOriginalInputsToGraph(kg, origin_inputs);
  }
  std::swap(new_switch_inputs[kCNodeSwitchTrue], new_switch_inputs[kCNodeSwitchFalse]);

  cur_node->set_inputs(new_switch_inputs);
  cur_node->set_abstract(nullptr);
  AnfAlgo::SetNodeAttr(kAttrChildGraph, MakeValue<std::vector<KernelGraphPtr>>(child_graphs), cur_node.get());
  MS_LOG(INFO) << "Succeed processing switch func " << cur_node->DebugString();
}

void AscendControlParser::RecurseSwitchLayer(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                                             const CNodePtr &next_node,
                                             const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "Process switch node " << cur_node->DebugString();

  if (cur_node->size() < kCNodeSwitchLayerLength) {
    MS_LOG(EXCEPTION) << "Inputs of apply node must more than " << kCNodeSwitchLayerLength;
  }

  std::vector<AnfNodePtr> branch_partial;
  for (size_t idx = kCNodeSwitchLayerBranch; idx < cur_node->inputs().size(); idx++) {
    branch_partial.emplace_back(cur_node->input(idx));
  }
  // 1 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  // 2 add depend relationship
  InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
  if (next_node != nullptr && next_node != kg->get_return()) {
    InsertControlDependToGraph(kg, NOT_NULL(back_label), NOT_NULL(next_node));
  }
  // 3 recurse sub graph
  const std::vector<AnfNodePtr> &origin_switch_inputs = cur_node->inputs();
  if (kCNodeSwitchCond >= origin_switch_inputs.size()) {
    MS_LOG(EXCEPTION) << "Index out of range:" << origin_switch_inputs.size() << ".";
  }
  std::vector<AnfNodePtr> new_switch_inputs = {
    std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
    origin_switch_inputs[kCNodeSwitchCond]};
  std::vector<KernelGraphPtr> child_graphs;
  for (size_t i = 0; i < branch_partial.size(); ++i) {
    // 3.1 branch kernel graph and args
    KernelGraphPtr branch_fg;
    std::vector<AnfNodePtr> origin_inputs;
    std::tie(branch_fg, origin_inputs) = ParsePartial(NOT_NULL(origin_switch_inputs[i + kCNodeSwitchLayerBranch]));
    child_graphs.push_back(branch_fg);
    // 3.2 recurse sub graph
    CNodePtr branch_label = ProcessKernelGraph(NOT_NULL(branch_fg), cur_node, back_label, memo);
    new_switch_inputs.push_back(branch_label);
    AttachOriginalInputsToGraph(kg, origin_inputs);
  }
  cur_node->set_inputs(new_switch_inputs);
  cur_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  // To adapt to the true and false branches of the switch, the sequence of the branches is reversed.
  std::reverse(child_graphs.begin(), child_graphs.end());
  AnfAlgo::SetNodeAttr(kAttrChildGraph, MakeValue<std::vector<KernelGraphPtr>>(child_graphs), cur_node.get());
  MS_LOG(INFO) << "Succeed processing switch layer " << cur_node->DebugString();
}

std::tuple<KernelGraphPtr, std::vector<AnfNodePtr>> AscendControlParser::ParsePartial(NotNull<AnfNodePtr> node) {
  if (!node.get()->isa<CNode>()) {
    if (IsValueNode<KernelGraph>(node)) {
      return {GetValueNode<KernelGraphPtr>(node), {}};
    }
    MS_LOG(EXCEPTION) << "Switch branches must be partial, node: " << node->DebugString();
  }
  // 2.1 branch kernel graph and args
  auto partial_cnode = utils::cast<CNodePtr>(node.get());
  MS_EXCEPTION_IF_NULL(partial_cnode);
  if (partial_cnode->size() < kCNodePartialLength) {
    MS_LOG(EXCEPTION) << "Inputs of partial node must more than " << kCNodePartialLength;
  }

  const auto &partial_inputs = partial_cnode->inputs();
  if (kCNodePartialFunc >= partial_inputs.size()) {
    MS_LOG(EXCEPTION) << "Index out of range:" << partial_inputs.size() << ".";
  }
  auto branch_kg = GetValueNode<KernelGraphPtr>(partial_inputs[kCNodePartialFunc]);
  return {branch_kg, std::vector<AnfNodePtr>(partial_inputs.begin() + kCNodePartialFunc + 1, partial_inputs.end())};
}

void AscendControlParser::InsertMultipleAssignToGraph(NotNull<KernelGraphPtr> from_graph, const AnfNodePtr &jump_node,
                                                      NotNull<AnfNodePtr> from, NotNull<AnfNodePtr> to) {
  std::vector<AnfNodePtr> from_outputs = AnfAlgo::GetAllOutput(from, {prim::kPrimTupleGetItem});
  std::vector<AnfNodePtr> to_outputs = AnfAlgo::GetAllOutput(to, {prim::kPrimTupleGetItem});
  MS_LOG(INFO) << "Insert multi-assign from [" << from->DebugString() << "] to [" << to->DebugString() << "]";
  if (from_outputs.size() != to_outputs.size()) {
    MS_LOG(EXCEPTION) << "From outputs size[" << from_outputs.size() << "] is not equal to to outputs size["
                      << to_outputs.size() << "]";
  }
  for (size_t i = 0; i < from_outputs.size(); i++) {
    auto assign_node = InsertAssignToGraph(from_graph, NOT_NULL(from_outputs[i]), NOT_NULL(to_outputs[i]));
    if (assign_node == nullptr) {
      continue;
    }
    const auto &from_graph_exe_order = from_graph->execution_order();
    if (jump_node == nullptr) {
      if (!from_graph_exe_order.empty()) {
        InsertControlDependToGraph(from_graph, NOT_NULL(*(from_graph_exe_order.rbegin())), NOT_NULL(assign_node));
      } else {
        InsertDependToGraph(from_graph, NOT_NULL(assign_node));
      }
      continue;
    }

    auto jump_node_iter = std::find(from_graph_exe_order.begin(), from_graph_exe_order.end(), jump_node);
    if (jump_node_iter == from_graph_exe_order.end()) {
      MS_LOG(EXCEPTION) << "Cannot find jump node " << jump_node->DebugString() << " in graph "
                        << from_graph->ToString();
    }
    // insert assign between jump_node -1 and jump_node
    while (jump_node_iter != from_graph_exe_order.begin()) {
      CNodePtr node = *(jump_node_iter - 1);
      if (AnfAlgo::GetGraphId(node.get()) == from_graph->graph_id()) {
        InsertControlDependToGraph(from_graph, NOT_NULL(*(jump_node_iter - 1)), NOT_NULL(assign_node));
        break;
      } else {
        jump_node_iter--;
      }
    }
    InsertControlDependToGraph(from_graph, NOT_NULL(assign_node), NOT_NULL(jump_node));
  }
}

AnfNodePtr AscendControlParser::InsertAssignToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> from,
                                                    NotNull<AnfNodePtr> to) {
  if (AnfAlgo::OutputAddrExist(from, 0) && AnfAlgo::OutputAddrExist(to, 0) &&
      AnfAlgo::GetOutputAddr(from, 0) == AnfAlgo::GetOutputAddr(to, 0)) {
    return nullptr;
  }
  if (from.get() == to.get()) {
    return nullptr;
  }
  MS_LOG(INFO) << "Insert assign to graph " << kg->ToString() << " from " << from->DebugString() << " to "
               << to->DebugString();
  // config inputs of assign node
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimAssign->name())), to, from};
  // generate a new cnode
  auto assign_node = kg->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(assign_node);
  assign_node->set_abstract(to->abstract());
  return assign_node;
}

std::vector<CNodePtr> AscendControlParser::RecurseGraph(NotNull<KernelGraphPtr> graph,
                                                        const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "Graph:" << graph->graph_id() << " start";
  if (memo->find(graph) != memo->end()) {
    return {};
  }
  memo->insert(graph.get());
  graph->SetExecOrderByDefault();
  std::vector<CNodePtr> cnodes = graph->execution_order();

  auto end_label_goto = graph->get_end_goto();
  if (cnodes.rbegin() != cnodes.rend() && *cnodes.rbegin() == end_label_goto) {
    cnodes.pop_back();
  }
  AnfAlgo::ReorderOptimizerExecList(NOT_NULL(&cnodes));
  if (end_label_goto != nullptr) {
    cnodes.push_back(end_label_goto);
  }

  std::vector<CNodePtr> execution_order;
  auto recurse_child_graph = [&](uint32_t index, uint32_t label_index, const CNodePtr &node) {
    KernelGraphPtr cur_child_graph;
    if (!CheckLabelIndex(index, label_index, node, &cur_child_graph)) {
      MS_LOG(EXCEPTION) << "Check label index fail";
    }
    MS_EXCEPTION_IF_NULL(cur_child_graph);
    auto child_execution_order = RecurseGraph(NOT_NULL(cur_child_graph), memo);
    execution_order.insert(execution_order.end(), child_execution_order.begin(), child_execution_order.end());
  };

  for (auto &node : cnodes) {
    uint32_t child_graph_index = 0;
    execution_order.push_back(node);
    if (node == graph->get_end_goto()) {
      continue;
    }
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSwitch)) {
      std::vector<uint32_t> label_switch_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(node, kAttrLabelSwitchList);
      for (auto iter = label_switch_list.rbegin(); iter != label_switch_list.rend(); ++iter) {
        recurse_child_graph(child_graph_index++, *iter, node);
      }
    } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelGoto)) {
      uint32_t label_index = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
      recurse_child_graph(child_graph_index, label_index, node);
    }
    // erase kAttrChildGraph after finish using
    if (AnfAlgo::HasNodeAttr(kAttrChildGraph, node)) {
      AnfAlgo::EraseNodeAttr(kAttrChildGraph, node);
    }
  }
  graph->set_execution_order(execution_order);
  return execution_order;
}

bool AscendControlParser::CheckLabelIndex(uint32_t index, uint32_t label_index, const CNodePtr &cur_label,
                                          KernelGraphPtr *cur_child_graph) {
  auto child_graphs = AnfAlgo::GetNodeAttr<std::vector<KernelGraphPtr>>(cur_label, kAttrChildGraph);
  // check index and child order size
  if (child_graphs.size() <= IntToSize(index)) {
    MS_LOG(EXCEPTION) << "Child graph index is wrong, current node " << cur_label->ToString() << " child graph size "
                      << child_graphs.size() << " goto index " << index;
  }
  *cur_child_graph = child_graphs[index];
  MS_EXCEPTION_IF_NULL(*cur_child_graph);

  // get start_label_set_index of child graph
  auto start_label_set = (*cur_child_graph)->get_start_label();
  uint32_t start_label_set_index = AnfAlgo::GetNodeAttr<uint32_t>(start_label_set, kAttrLabelIndex);
  if (label_index != start_label_set_index) {
    MS_EXCEPTION_IF_NULL(cur_label);
    MS_EXCEPTION_IF_NULL(start_label_set);
    MS_LOG(WARNING) << cur_label->DebugString() << " index " << label_index << " but " << start_label_set->DebugString()
                    << " index " << start_label_set_index;
    return false;
  } else {
    return true;
  }
}

void AscendControlParser::ReferenceCounter::AddReadCount(const AnfNodePtr &key, int64_t num) {
  auto iter = count_.find(key);
  if (iter != count_.end()) {
    iter->second.first += num;
  } else {
    count_[key] = {num, 0};
  }
}

void AscendControlParser::ReferenceCounter::AddWriteCount(const AnfNodePtr &key, int64_t num) {
  auto iter = count_.find(key);
  if (iter != count_.end()) {
    iter->second.second += num;
  } else {
    count_[key] = {0, num};
  }
}

void AscendControlParser::ReferenceCounter::EraseElem(const AnfNodePtr &key) { count_.erase(key); }

bool AscendControlParser::ReferenceCounter::HasValidElem() const {
  auto it = std::find_if(count_.begin(), count_.end(),
                         [this](const std::pair<AnfNodePtr, std::pair<uint32_t, uint32_t>> &p) -> bool {
                           auto &[read, written] = p.second;
                           return predicate_(read, written);
                         });
  return it != count_.end();
}

std::tuple<AnfNodePtr, int64_t, int64_t> AscendControlParser::ReferenceCounter::GetOneValidElem() const {
  auto it = std::find_if(count_.begin(), count_.end(),
                         [this](const std::pair<AnfNodePtr, std::pair<uint32_t, uint32_t>> &p) -> bool {
                           auto &[read, written] = p.second;
                           return predicate_(read, written);
                         });
  if (it == count_.end()) {
    MS_LOG(EXCEPTION) << "No valid parameter.";
  }
  return {it->first, it->second.first, it->second.second};
}
}  // namespace session
}  // namespace mindspore
