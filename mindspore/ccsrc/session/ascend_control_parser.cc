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

#include <utility>
#include <memory>
#include "session/ascend_control_parser.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/union_find_set.h"

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

namespace mindspore {
namespace session {
static void InitUnionFindSet(NotNull<KernelGraphPtr> kg, const NotNull<UnionFindSet<AnfNodePtr> *> union_find_set,
                             const NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(kg.get()) != memo->end()) {
    return;
  }
  memo->insert(kg.get());

  const std::map<AnfNodePtr, std::set<AnfNodePtr>> &real_inputs = kg->real_inputs();
  for (auto &iter : real_inputs) {
    auto &para = iter.first;
    if (para->isa<Parameter>()) {
      union_find_set->Add(para);
    }
    for (auto &arg : iter.second) {
      if (!arg->isa<Parameter>()) {
        continue;
      }
      union_find_set->Add(arg);
    }
  }
  for (auto &child : kg->child_graph_order()) {
    InitUnionFindSet(NOT_NULL(child), union_find_set, memo);
  }
}

static void UnionParentParameter(NotNull<KernelGraphPtr> kg, const NotNull<UnionFindSet<AnfNodePtr> *> union_find_set,
                                 const NotNull<std::set<KernelGraphPtr> *> memo) {
  if (memo->find(kg.get()) != memo->end()) {
    return;
  }
  memo->insert(kg.get());

  const std::map<AnfNodePtr, std::set<AnfNodePtr>> &real_inputs = kg->real_inputs();
  for (auto &iter : real_inputs) {
    auto &para = iter.first;
    for (auto &arg : iter.second) {
      if (!arg->isa<Parameter>()) {
        continue;
      }
      union_find_set->Union(arg, para);
    }
  }
  for (auto &child : kg->child_graph_order()) {
    UnionParentParameter(NOT_NULL(child), union_find_set, memo);
  }
}

static UnionFindSet<AnfNodePtr> MakeUnionFindSet(NotNull<KernelGraphPtr> root_kg) {
  UnionFindSet<AnfNodePtr> result;
  std::set<KernelGraphPtr> memo;
  InitUnionFindSet(root_kg, NOT_NULL(&result), NOT_NULL(&memo));
  memo.clear();
  UnionParentParameter(root_kg, NOT_NULL(&result), NOT_NULL(&memo));
  return result;
}

static void RecursiveReplaceNode(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> main_parameter,
                                 const std::set<AnfNodePtr> &parameter_reuse_set,
                                 const NotNull<std::set<KernelGraphPtr> *> memo) {
  if (parameter_reuse_set.empty()) {
    MS_LOG(EXCEPTION) << "parameter_reuse_set is empty.";
  }
  if (memo->find(kg.get()) != memo->end()) {
    return;
  }
  memo->insert(kg.get());

  for (auto &para : parameter_reuse_set) {
    if (para == main_parameter.get()) {
      continue;
    }
    MS_LOG(INFO) << "Replace " << para->DebugString() << " of graph " << AnfAlgo::GetGraphId(para.get()) << " to "
                 << main_parameter->DebugString() << " of graph " << AnfAlgo::GetGraphId(main_parameter.get().get());
    kg->ReplaceNode(NOT_NULL(para), main_parameter);
  }

  for (auto &child : kg->child_graph_order()) {
    RecursiveReplaceNode(NOT_NULL(child), main_parameter, parameter_reuse_set, memo);
  }
}

static void ReuseParameter(NotNull<KernelGraphPtr> root_kg, NotNull<UnionFindSet<AnfNodePtr> *> parameter_set) {
  auto parameter_reuse_sets = parameter_set->GetSets();
  for (auto &[key, parameter_reuse_set] : parameter_reuse_sets) {
    if (parameter_reuse_set.size() <= 1) {
      continue;
    }

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

    std::set<KernelGraphPtr> memo;
    RecursiveReplaceNode(root_kg, NOT_NULL(main_parameter), parameter_reuse_set, NOT_NULL(&memo));
  }
}

CNodePtr GetNextRealKernel(const std::vector<CNodePtr> &list, size_t start) {
  for (size_t i = start; i < list.size() - 1; ++i) {
    if (!IsPrimitiveCNode(list[i], prim::kPrimPartial) && AnfAlgo::IsRealKernel(list[i])) {
      return list[i];
    }
  }
  return nullptr;
}

void AscendControlParser::LinkGraph(NotNull<KernelGraphPtr> kg) {
  std::set<KernelGraphPtr> memo;
  (void)ProcessKernelGraph(kg, nullptr, nullptr, NOT_NULL(&memo));
  std::map<uint32_t, KernelGraphPtr> graph_id_map;
  for (auto &g : memo) {
    if (graph_id_map.find(g->graph_id()) != graph_id_map.end()) {
      MS_LOG(EXCEPTION) << "Two graph has same graph id " << g->graph_id()
                        << ", graph: " << graph_id_map[g->graph_id()]->ToString() << " " << g->ToString();
    }
    graph_id_map[g->graph_id()] = g;
  }
  // Make UnionFindSet
  UnionFindSet<AnfNodePtr> parameter_set = MakeUnionFindSet(kg);
  // Reuse Parameter
  ReuseParameter(kg, NOT_NULL(&parameter_set));
  // Insert Assign
  ChildGraphDataAssign(graph_id_map);
}

void AscendControlParser::ExecutorValidate(NotNull<KernelGraphPtr> root_graph) {
  std::set<KernelGraphPtr> memo;
  (void)RecurseGraph(root_graph, NOT_NULL(&memo));
}

void AscendControlParser::ChildGraphDataAssign(const std::map<uint32_t, KernelGraphPtr> &graph_id_map) {
  for (auto &iter : graph_id_map) {
    auto &kg = iter.second;
    MS_EXCEPTION_IF_NULL(kg);
    auto real_inputs = kg->real_inputs();
    for (auto &it : real_inputs) {
      auto &parameter = it.first;
      auto &args = it.second;
      for (auto &arg : args) {
        MS_EXCEPTION_IF_NULL(arg);
        if (arg->isa<Parameter>()) {
          MS_LOG(DEBUG) << "Parameter should be reused, no need insert assign, parameter: " << parameter->DebugString()
                        << ", arg:" << arg->DebugString();
          continue;
        }
        auto target_graph_iter = graph_id_map.find(AnfAlgo::GetGraphId(arg.get()));
        if (target_graph_iter == graph_id_map.end()) {
          MS_LOG(EXCEPTION) << "Graph id " << AnfAlgo::GetGraphId(arg.get()) << " not found.";
        }
        InsertAssignToGraph(NOT_NULL(target_graph_iter->second), NOT_NULL(arg), NOT_NULL(parameter));
      }
    }
  }
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
  if (nodes.empty()) {
    MS_LOG(EXCEPTION) << "KernelGraph " << kg->ToString() << " has no cnodes!";
  }
  // 4. insert first_label
  CNodePtr start_label;
  if (last_node != nullptr && last_label != nullptr) {
    start_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
    MS_LOG(INFO) << "Insert start label " << start_label->DebugString() << " to " << kg->ToString();
    kg->set_start_label(start_label);
  } else {
    // no goto node will jump to start label of root graph, so return a fake label
    start_label = std::make_shared<CNode>(std::vector<AnfNodePtr>(), FuncGraphPtr(nullptr));
  }

  // 5. traverse
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto &cnode = nodes[i];
    if (cnode->size() < kCNodePrim + 1) {
      MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
    }
    AnfNodePtr fn = cnode->input(kCNodePrim);
    if (!IsPrimitive(fn, prim::kPrimCall) || cnode->size() < kCNodeCallArg + 1) {
      MS_LOG(DEBUG) << "continue node " << cnode->DebugString();
      continue;
    }
    AnfNodePtr arg = cnode->input(kCNodeCallArg);
    if (IsValueNode<KernelGraph>(arg)) {
      RecurseCall(kg, NOT_NULL(cnode), GetNextRealKernel(nodes, i + 1), memo);
    } else if (!arg->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Unknown type call node " << cnode->DebugString();
    } else if (IsPrimitiveCNode(arg->cast<CNodePtr>(), prim::kPrimSwitch)) {
      auto arg_cnode = arg->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(arg_cnode);
      cnode->set_inputs(arg_cnode->inputs());
      RecurseSwitch(kg, NOT_NULL(cnode), GetNextRealKernel(nodes, i + 1), memo);
    } else if (IsPrimitiveCNode(arg->cast<CNodePtr>(), prim::kPrimSwitchLayer)) {
      auto arg_cnode = arg->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(arg_cnode);
      cnode->set_inputs(arg_cnode->inputs());
      RecurseSwitchLayer(kg, NOT_NULL(cnode), GetNextRealKernel(nodes, i + 1), memo);
    }
  }

  MS_LOG(INFO) << "End KernelGraph process: " << kg->ToString();
  return NOT_NULL(start_label);
}

void AscendControlParser::InsertDependToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> attch_node) {
  auto return_node = kg->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                    return_node->input(1), attch_node.get()};
  auto depend_node = kg->NewCNode(inputs);
  return_node->set_input(1, depend_node);
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
  auto origin_return = kg->get_return();
  const std::vector<AnfNodePtr> &origin_return_inputs = origin_return->inputs();
  // if entry graph, replace return with make_tuple
  if (from_graph_call_node == nullptr || last_label == nullptr) {
    MS_LOG(INFO) << kg->ToString() << " is entry graph.";
    std::vector<AnfNodePtr> make_tuple_inputs = {std::make_shared<ValueNode>(prim::kPrimMakeTuple)};
    make_tuple_inputs.insert(make_tuple_inputs.end(), origin_return_inputs.begin() + 1, origin_return_inputs.end());
    auto make_tuple = kg->NewCNode(make_tuple_inputs);
    origin_return->set_inputs({origin_return->input(kCNodePrim), make_tuple});
  } else {
    // else replace return with label_goto
    auto label_goto =
      kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelGotoOpName)), last_label});
    MS_LOG(INFO) << "Insert end goto " << label_goto->DebugString() << " to " << kg->ToString();
    kg->set_end_goto(label_goto);
  }
}

void AscendControlParser::RecurseCall(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node, const CNodePtr &next_node,
                                      const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "process call func " << cur_node->DebugString();

  // 1 get kernel graph
  const std::vector<AnfNodePtr> &origin_inputs = cur_node->inputs();
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
  new_inputs.insert(new_inputs.end(), origin_inputs.begin(), origin_inputs.end());
  cur_node->set_inputs(new_inputs);
  cur_node->set_abstract(nullptr);
  MS_LOG(INFO) << "success process call func " << cur_node->DebugString();
}

void AscendControlParser::RecurseSwitch(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                                        const CNodePtr &next_node, const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "process switch node " << cur_node->DebugString();

  if (cur_node->size() < kCNodeSwitchLength) {
    MS_LOG(EXCEPTION) << "Inputs of apply node must more than " << kCNodeSwitchLength;
  }
  // 1 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  MS_LOG(INFO) << "Insert back label " << back_label->DebugString() << " to " << kg->ToString() << " switch node "
               << cur_node->DebugString();
  // 2 add depend relationship
  InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
  if (next_node != nullptr && next_node != kg->get_return()) {
    InsertControlDependToGraph(kg, NOT_NULL(back_label), NOT_NULL(next_node));
  }
  // 3 recurse sub graph
  const std::vector<AnfNodePtr> &origin_switch_inputs = cur_node->inputs();
  std::vector<AnfNodePtr> new_switch_inputs = {
    std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
    origin_switch_inputs[kCNodeSwitchCond]};
  for (size_t i = kCNodeSwitchCond + 1; i < kCNodeSwitchLength; ++i) {
    // 3.1 branch kernel graph and args
    KernelGraphPtr branch_fg;
    std::tie(std::ignore, branch_fg) = ParsePartial(NOT_NULL(origin_switch_inputs[i]));
    // 3.2 recurse sub graph
    CNodePtr branch_label = ProcessKernelGraph(NOT_NULL(branch_fg), cur_node, back_label, memo);
    new_switch_inputs.push_back(branch_label);
  }
  std::swap(new_switch_inputs[kCNodeSwitchTrue], new_switch_inputs[kCNodeSwitchFalse]);

  new_switch_inputs.insert(new_switch_inputs.end(), origin_switch_inputs.begin(), origin_switch_inputs.end());
  cur_node->set_inputs(new_switch_inputs);
  cur_node->set_abstract(nullptr);
  MS_LOG(INFO) << "success process switch func " << cur_node->DebugString();
}

void AscendControlParser::RecurseSwitchLayer(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                                             const CNodePtr &next_node,
                                             const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "process switch node " << cur_node->DebugString();

  if (cur_node->size() < kCNodeSwitchLayerLength) {
    MS_LOG(EXCEPTION) << "Inputs of apply node must more than " << kCNodeSwitchLayerLength;
  }

  auto branch_tuple = cur_node->input(kCNodeSwitchLayerBranch);
  MS_EXCEPTION_IF_NULL(branch_tuple);
  if (!branch_tuple->isa<CNode>()) {
    MS_LOG(EXCEPTION) << branch_tuple->DebugString() << " is not a CNode";
  }
  const std::vector<AnfNodePtr> &branch_partial = utils::cast<CNodePtr>(branch_tuple)->inputs();
  // 1 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  // 2 add depend relationship
  InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
  if (next_node != nullptr && next_node != kg->get_return()) {
    InsertControlDependToGraph(kg, NOT_NULL(back_label), NOT_NULL(next_node));
  }
  // 3 recurse sub graph
  const std::vector<AnfNodePtr> &origin_switch_inputs = cur_node->inputs();
  std::vector<AnfNodePtr> new_switch_inputs = {
    std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
    origin_switch_inputs[kCNodeSwitchCond]};
  for (size_t i = 0; i < branch_partial.size(); ++i) {
    // 3.1 branch kernel graph and args
    KernelGraphPtr branch_fg;
    std::tie(std::ignore, branch_fg) = ParsePartial(NOT_NULL(origin_switch_inputs[i]));
    // 3.2 recurse sub graph
    CNodePtr branch_label = ProcessKernelGraph(NOT_NULL(branch_fg), cur_node, back_label, memo);
    new_switch_inputs.push_back(branch_label);
  }
  new_switch_inputs.insert(new_switch_inputs.end(), branch_partial.begin(), branch_partial.end());
  cur_node->set_inputs(new_switch_inputs);
  cur_node->set_abstract(nullptr);
  MS_LOG(INFO) << "success process switch layer " << cur_node->DebugString();
}

std::tuple<CNodePtr, KernelGraphPtr> AscendControlParser::ParsePartial(NotNull<AnfNodePtr> node) {
  if (!node.get()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Switch branches must be partial, node: " << node->DebugString();
  }
  // 2.1 branch kernel graph and args
  auto partial_cnode = utils::cast<CNodePtr>(node.get());
  if (partial_cnode->size() < kCNodePartialLength) {
    MS_LOG(EXCEPTION) << "Inputs of partial node must more than " << kCNodePartialLength;
  }

  auto partial_inputs = partial_cnode->inputs();
  auto branch_kg = GetValueNode<KernelGraphPtr>(partial_inputs[kCNodePartialFunc]);
  return {partial_cnode, branch_kg};
}

void AscendControlParser::InsertAssignToGraph(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> from,
                                              NotNull<AnfNodePtr> to) {
  if (AnfAlgo::OutputAddrExist(from, 0) && AnfAlgo::OutputAddrExist(to, 0) &&
      AnfAlgo::GetOutputAddr(from, 0) == AnfAlgo::GetOutputAddr(to, 0)) {
    return;
  }
  if (from.get() == to.get()) {
    return;
  }
  MS_LOG(INFO) << "Insert assign to graph " << kg->ToString() << " from " << from->DebugString() << " to "
               << to->DebugString();
  // config inputs of assign node
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimAssign->name())), to, from};
  // generate a new cnode
  auto assign_node = kg->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(assign_node);
  assign_node->set_abstract(to->abstract());
  // append the assign at the end of from graph
  InsertDependToGraph(kg, NOT_NULL(assign_node));
}

std::vector<CNodePtr> AscendControlParser::RecurseGraph(NotNull<KernelGraphPtr> graph,
                                                        const NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "graph:" << graph->graph_id() << " start";
  if (memo->find(graph) != memo->end()) {
    return {};
  }
  memo->insert(graph.get());
  graph->SetExecOrderByDefault();
  const std::vector<CNodePtr> &cnodes = graph->execution_order();

  std::vector<CNodePtr> execution_order;
  uint32_t child_order_index = 0;

  for (auto &node : cnodes) {
    execution_order.push_back(node);
    if (node == graph->get_end_goto()) {
      continue;
    }
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSwitch)) {
      std::vector<uint32_t> label_switch_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(node, kAttrLabelSwitchList);
      for (auto iter = label_switch_list.rbegin(); iter != label_switch_list.rend(); ++iter) {
        if (!CheckLabelIndex(child_order_index, *iter, node, graph)) {
          MS_LOG(EXCEPTION) << "Check label index fail";
        }
        auto child_graph = graph->child_graph_order()[child_order_index++];
        auto child_execution_order = RecurseGraph(NOT_NULL(child_graph), memo);
        execution_order.insert(execution_order.end(), child_execution_order.begin(), child_execution_order.end());
      }
    } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelGoto)) {
      uint32_t label_index = AnfAlgo::GetNodeAttr<uint32_t>(node, kAttrLabelIndex);
      if (!CheckLabelIndex(child_order_index, label_index, node, graph)) {
        MS_LOG(EXCEPTION) << "Check label index fail";
      }
      auto child_graph = graph->child_graph_order()[child_order_index++];
      auto child_execution_order = RecurseGraph(NOT_NULL(child_graph), memo);
      execution_order.insert(execution_order.end(), child_execution_order.begin(), child_execution_order.end());
    }
  }
  graph->set_execution_order(execution_order);
  graph->PrintGraphExecuteOrder();
  return execution_order;
}

bool AscendControlParser::CheckLabelIndex(uint32_t order_index, uint32_t label_index, const CNodePtr &cur_label,
                                          NotNull<KernelGraphPtr> graph) {
  const std::vector<std::shared_ptr<KernelGraph>> &child_graph_order = graph->child_graph_order();
  // check index and child order size
  if (child_graph_order.size() <= IntToSize(order_index)) {
    MS_LOG(EXCEPTION) << "Child graph order is wrong, graph " << graph->ToString() << " child graph size "
                      << child_graph_order.size() << " goto index " << order_index;
  }
  auto child_graph = child_graph_order[order_index];
  MS_EXCEPTION_IF_NULL(child_graph);

  // get start_label_set_index of child graph
  auto start_label_set = child_graph->get_start_label();
  uint32_t start_label_set_index = AnfAlgo::GetNodeAttr<uint32_t>(start_label_set, kAttrLabelIndex);
  if (label_index != start_label_set_index) {
    MS_LOG(WARNING) << cur_label->DebugString() << " index " << label_index << " but " << start_label_set->DebugString()
                    << " index " << start_label_set_index << " current child graph order : " << order_index;
    return false;
  } else {
    return true;
  }
}

void AscendControlParser::UpdateChildGraphOrder(NotNull<KernelGraphPtr> kg) {
  MS_LOG(INFO) << "graph id:" << kg->graph_id();
  kg->SetExecOrderByDefault();
  auto call_nodes = kg->FindNodeByPrimitive(std::make_shared<Primitive>(prim::kPrimCall->name()));
  std::vector<KernelGraphPtr> child_graph_order;
  for (auto &call_node : call_nodes) {
    MS_EXCEPTION_IF_NULL(call_node);
    auto call_child_graphs = AnfAlgo::GetCallNodeKernelGraph(call_node->cast<CNodePtr>());
    for (const auto &child_graph : call_child_graphs) {
      MS_EXCEPTION_IF_NULL(child_graph);
      if (child_graph != kg->parent_graph()) {
        child_graph->set_parent_graph(kg.get());
      }
      child_graph_order.push_back(child_graph);
    }
  }
  for (size_t i = 0; i < child_graph_order.size(); i++) {
    MS_LOG(INFO) << "child graph[" << i << "][id:" << child_graph_order[i]->graph_id() << "]";
  }
  kg->set_child_graph_order(child_graph_order);
}
}  // namespace session
}  // namespace mindspore
