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

namespace mindspore {
namespace session {

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
          MS_LOG(INFO) << "Parameter should be reused, no need insert assign, parameter: " << parameter->DebugString()
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

void AscendControlParser::LinkGraph(NotNull<KernelGraphPtr> kg) {
  std::set<KernelGraphPtr> memo;
  ProcessKernelGraph(kg, nullptr, nullptr, NOT_NULL(&memo));
  std::map<uint32_t, KernelGraphPtr> graph_id_map;
  for (auto &g : memo) {
    if (graph_id_map.find(g->graph_id()) != graph_id_map.end()) {
      MS_LOG(EXCEPTION) << "Two graph has same graph id " << g->graph_id()
                        << ", graph: " << graph_id_map[g->graph_id()]->ToString() << " " << g->ToString();
    }
    graph_id_map[g->graph_id()] = g;
  }
  ChildGraphDataAssign(graph_id_map);
}

CNodePtr AscendControlParser::GetNextRealKernel(std::vector<CNodePtr> list, size_t start) {
  for (size_t i = start; i < list.size() - 1; ++i) {
    if (!IsPrimitiveCNode(list[i], prim::kPrimPartial) && AnfAlgo::IsRealKernel(list[i])) {
      return list[i];
    }
  }
  return nullptr;
}

NotNull<CNodePtr> AscendControlParser::ProcessKernelGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &last_node,
                                                          const CNodePtr &last_label,
                                                          NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "Start process KernelGraph " << kg->ToString();

  // 1. recursive condition
  if (memo->find(kg) != memo->end()) {
    MS_LOG(INFO) << "KernelGraph has beed processed: " << kg->ToString();
    return NOT_NULL(kg->get_start_label());
  }
  memo->insert(kg.get());

  // 2. args replace placeholder
  LinkParentGraph(kg, last_node, last_label, memo);

  // 3. topological sort
  kg->SetExecOrderByDefault();
  std::vector<CNodePtr> nodes = kg->execution_order();
  if (nodes.empty()) {
    MS_LOG(EXCEPTION) << "KernelGraph " << kg->ToString() << " has no cnodes!";
  }
  // 4. insert first_label
  auto start_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  MS_LOG(INFO) << "Insert start label " << start_label->DebugString() << " to " << kg->ToString();
  kg->set_start_label(start_label);
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
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("depend"))};
  auto return_node = kg->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  inputs.push_back(return_node->input(1));
  inputs.push_back(attch_node.get());
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
                                          const CNodePtr &last_label, NotNull<std::set<KernelGraphPtr> *> memo) {
  auto origin_return = kg->get_return();
  std::vector<AnfNodePtr> origin_return_inputs = origin_return->inputs();
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
                                      NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "process call func " << cur_node->DebugString();

  // 1 get kernel graph
  auto origin_inputs = cur_node->inputs();
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
                                        const CNodePtr &next_node, NotNull<std::set<KernelGraphPtr> *> memo) {
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
  auto origin_switch_inputs = cur_node->inputs();
  std::vector<AnfNodePtr> new_switch_inputs = {
    std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
    origin_switch_inputs[kCNodeSwitchCond]};
  for (size_t i = kCNodeSwitchCond + 1; i < kCNodeSwitchLength; ++i) {
    // 3.1 branch kernel graph and args
    CNodePtr partial;
    KernelGraphPtr branch_fg;
    std::tie(partial, branch_fg) = ParsePartial(NOT_NULL(origin_switch_inputs[i]));
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
                                             const CNodePtr &next_node, NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "process switch node " << cur_node->DebugString();

  if (cur_node->size() < kCNodeSwitchLayerLength) {
    MS_LOG(EXCEPTION) << "Inputs of apply node must more than " << kCNodeSwitchLayerLength;
  }

  auto branch_tuple = cur_node->input(kCNodeSwitchLayerBranch);
  MS_EXCEPTION_IF_NULL(branch_tuple);
  if (!branch_tuple->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Inputs of apply node must more than " << kCNodeSwitchLayerLength;
  }
  auto branch_partial = utils::cast<CNodePtr>(branch_tuple)->inputs();
  // 1 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  // 2 add depend relationship
  InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
  if (next_node != nullptr && next_node != kg->get_return()) {
    InsertControlDependToGraph(kg, NOT_NULL(back_label), NOT_NULL(next_node));
  }
  // 3 recurse sub graph
  auto origin_switch_inputs = cur_node->inputs();
  std::vector<AnfNodePtr> new_switch_inputs = {
    std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
    origin_switch_inputs[kCNodeSwitchCond]};
  for (size_t i = 0; i < branch_partial.size(); ++i) {
    // 3.1 branch kernel graph and args
    CNodePtr partial;
    KernelGraphPtr branch_fg;
    std::tie(partial, branch_fg) = ParsePartial(NOT_NULL(origin_switch_inputs[i]));
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
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>("Assign")), to, from};
  // generate a new cnode
  auto assign_node = kg->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(assign_node);
  assign_node->set_abstract(to->abstract());
  // append the assign at the end of from graph
  InsertDependToGraph(kg, NOT_NULL(assign_node));
}

NotNull<AnfNodePtr> AscendControlParser::GetRealInput(NotNull<KernelGraphPtr> from_graph,
                                                      NotNull<KernelGraphPtr> to_graph, NotNull<AnfNodePtr> param) {
  std::set<AnfNodePtr> args_list = to_graph->GetRealInput(param);
  for (auto arg : args_list) {
    if (arg->func_graph() == from_graph.get()) {
      return NOT_NULL(arg);
    }
  }
  MS_LOG(EXCEPTION) << to_graph->ToString() << " input " << param->DebugString() << " not from "
                    << from_graph->ToString();
}

void AscendControlParser::LinkArgsToParam(NotNull<KernelGraphPtr> to_graph, NotNull<KernelGraphPtr> target_graph,
                                          NotNull<AnfNodePtr> arg, NotNull<AnfNodePtr> param) {
  if (IsPrimitiveCNode(arg, prim::kPrimMakeTuple) && IsPrimitiveCNode(param, prim::kPrimMakeTuple)) {
    MS_LOG(INFO) << "Arg " << arg->DebugString() << " Param " << param->DebugString() << " is a tuple";
    CNodePtr cnode_arg = arg.get()->cast<CNodePtr>();
    CNodePtr cnode_param = param.get()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_arg);
    MS_EXCEPTION_IF_NULL(cnode_param);
    if (cnode_arg->size() != cnode_param->size()) {
      MS_LOG(EXCEPTION) << "Arg " << arg->DebugString() << " size " << cnode_arg->size() << " but Param "
                        << param->DebugString() << " size " << cnode_param->size();
    }

    for (size_t i = 1; i < cnode_param->size(); ++i) {
      LinkArgsToParam(to_graph, target_graph, NOT_NULL(cnode_arg->input(i)), NOT_NULL(cnode_param->input(i)));
    }
  } else if (arg->isa<CNode>()) {
    InsertAssignToGraph(target_graph, arg, param);
  } else {
    MS_LOG(EXCEPTION) << "Arg " << arg->DebugString() << " Param " << param->DebugString() << " unknown type.";
  }
}

void AscendControlParser::ExecutorValidate(NotNull<KernelGraphPtr> root_graph) {
  std::set<KernelGraphPtr> memo;
  (void)RecurseGraph(nullptr, nullptr, root_graph, NOT_NULL(&memo));
}

std::vector<CNodePtr> AscendControlParser::RecurseGraph(const CNodePtr &cur_label_goto, const CNodePtr &end_label_goto,
                                                        NotNull<KernelGraphPtr> graph,
                                                        NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "graph:" << graph->graph_id() << " start";
  auto print_vector = [&](std::vector<CNodePtr> vec) -> void {
    MS_LOG(INFO) << "graph:" << graph->graph_id() << "execution order";
    for (size_t i = 0; i < vec.size(); i++) {
      MS_LOG(INFO) << "[" << i << "][" << vec[i]->DebugString() << "]";
    }
  };
  if (memo->find(graph) != memo->end()) {
    return {};
  }
  memo->insert(graph.get());

  graph->SetExecOrderByDefault();

  std::vector<CNodePtr> cnodes = graph->execution_order();
  std::map<uint32_t, CNodePtr> label_map;
  std::map<CNodePtr, std::vector<uint32_t>> label_switch_map;
  std::tie(label_map, label_switch_map) = GetLabelNode(cnodes);
  std::vector<CNodePtr> execution_order;

  for (auto &node : cnodes) {
    execution_order.push_back(node);
    if (node == graph->get_end_goto()) {
      continue;
    }

    auto label_iter =
      std::find_if(label_map.begin(), label_map.end(),
                   [node](const std::map<uint32_t, CNodePtr>::value_type iter) { return iter.second == node; });
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelGoto)) {
      if (!CheckLabelIndex(label_iter->first, 0, label_iter->second, graph)) {
        MS_LOG(EXCEPTION) << "Check label index fail";
      }
      auto child_graph = graph->child_graph_order()[label_iter->first];
      if (child_graph == graph->parent_graph()) {
        continue;
      }
      std::map<uint32_t, CNodePtr> child_label_map;
      std::tie(child_label_map, std::ignore) = GetLabelNode(child_graph->execution_order());
      auto child_execution_order =
        RecurseGraph(child_label_map.begin()->second, child_label_map.rbegin()->second, NOT_NULL(child_graph), memo);
      execution_order.insert(execution_order.end(), child_execution_order.begin(), child_execution_order.end());
    } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSwitch)) {
      std::vector<uint32_t> label_list = label_switch_map.find(node)->second;
      std::reverse(label_list.begin(), label_list.end());
      for (size_t i = 0; i < label_list.size(); ++i) {
        if (!CheckLabelIndex(label_iter->first + i, label_list[i], label_iter->second, graph)) {
          MS_LOG(EXCEPTION) << "Check label index fail";
        }
        auto child_graph = graph->child_graph_order()[label_iter->first + i];
        if (child_graph == graph->parent_graph()) {
          continue;
        }
        std::map<uint32_t, CNodePtr> child_label_map;
        std::tie(child_label_map, std::ignore) = GetLabelNode(child_graph->execution_order());
        auto child_execution_order =
          RecurseGraph(child_label_map.begin()->second, child_label_map.rbegin()->second, NOT_NULL(child_graph), memo);
        execution_order.insert(execution_order.end(), child_execution_order.begin(), child_execution_order.end());
      }
    }
  }
  graph->set_execution_order(execution_order);
  print_vector(graph->execution_order());
  return execution_order;
}

bool AscendControlParser::CheckLabelIndex(uint32_t order_index, uint32_t label_index, const CNodePtr &cur_label,
                                          NotNull<KernelGraphPtr> graph) {
  // check index and child order size
  if (graph->child_graph_order().size() <= static_cast<size_t>(order_index)) {
    MS_LOG(EXCEPTION) << "Child graph order is wrong, graph " << graph->ToString() << " child graph size "
                      << graph->child_graph_order().size() << " goto index " << order_index;
  }

  if (AnfAlgo::CheckPrimitiveType(cur_label, prim::kPrimLabelGoto)) {
    // check label_goto and start_label in child graph
    if (!AnfAlgo::HasNodeAttr(kAttrLabelIndex, cur_label)) {
      MS_LOG(EXCEPTION) << "LabelSetKernel has no attr label_index";
    }
    auto primitive = AnfAlgo::GetCNodePrimitive(cur_label);
    MS_EXCEPTION_IF_NULL(primitive);
    uint32_t label_goto_index = GetValue<uint32_t>(primitive->GetAttr(kAttrLabelIndex));
    label_index = label_goto_index;
  }
  // get start_label_set_index of child graph
  auto child_graph = graph->child_graph_order()[order_index];
  MS_EXCEPTION_IF_NULL(child_graph);
  auto start_label_set = child_graph->get_start_label();
  if (!AnfAlgo::HasNodeAttr(kAttrLabelIndex, start_label_set)) {
    MS_LOG(EXCEPTION) << "LabelSetKernel has no attr label_index";
  }
  auto start_primitive = AnfAlgo::GetCNodePrimitive(start_label_set);
  MS_EXCEPTION_IF_NULL(start_primitive);
  uint32_t start_label_set_index = GetValue<uint32_t>(start_primitive->GetAttr(kAttrLabelIndex));
  if (label_index != start_label_set_index) {
    MS_LOG(WARNING) << cur_label->DebugString() << " index " << label_index << " but " << start_label_set->DebugString()
                    << " index " << start_label_set_index << " current child graph order : " << order_index;
    return false;
  }
  return true;
}

std::tuple<std::map<uint32_t, CNodePtr>, std::map<CNodePtr, std::vector<uint32_t>>> AscendControlParser::GetLabelNode(
  const std::vector<CNodePtr> &nodes) {
  std::map<uint32_t, CNodePtr> label_map;
  std::map<CNodePtr, std::vector<uint32_t>> label_switch_map;
  // record child graph
  uint32_t index = 0;
  for (auto &node : nodes) {
    if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelGoto)) {
      label_map[index] = node;
      ++index;
    } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimLabelSwitch)) {
      if (!AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, node)) {
        MS_LOG(EXCEPTION) << "LabelSwitchKernel has no attr label_switch_list";
      }
      auto primitive = AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(primitive);
      std::vector<uint32_t> label_list = GetValue<std::vector<uint32_t>>(primitive->GetAttr(kAttrLabelSwitchList));
      label_switch_map.insert({node, label_list});
      for (size_t i = 0; i < label_list.size(); ++i) {
        label_map[index] = node;
        ++index;
      }
    }
  }
  return {label_map, label_switch_map};
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
