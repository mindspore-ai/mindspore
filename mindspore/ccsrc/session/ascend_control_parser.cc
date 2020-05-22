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

static VectorRef GetCallArgs(std::vector<AnfNodePtr>::iterator iter_begin, std::vector<AnfNodePtr>::iterator iter_end) {
  VectorRef call_args;
  for (auto iter = iter_begin; iter != iter_end; ++iter) {
    if (utils::isa<ValueNode>(*iter)) {
      call_args.push_back(GetValueNode(*iter));
    } else {
      call_args.push_back(*iter);
    }
  }
  return call_args;
}

void AscendControlParser::LinkGraph(NotNull<KernelGraphPtr> kg) {
  std::set<KernelGraphPtr> memo;
  ProcessKernelGraph(kg, nullptr, nullptr, {}, NOT_NULL(&memo));
}

NotNull<CNodePtr> AscendControlParser::ProcessKernelGraph(NotNull<KernelGraphPtr> kg, const CNodePtr &last_node,
                                                          const CNodePtr &last_label, const VectorRef &args,
                                                          NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "Start process KernelGraph " << kg->ToString();
  // 0. recursive condition
  if (memo->find(kg) != memo->end()) {
    MS_LOG(INFO) << "KernelGraph has beed processed: " << kg->ToString();
    return NOT_NULL(kg->get_start_label());
  }

  // 2. args replace placeholder
  LinkParentGraph(kg, last_node, last_label, args);
  // 3. topological sort
  std::vector<CNodePtr> nodes = GetCNodes(TopoSort(kg->get_return()));
  if (nodes.empty()) {
    MS_LOG(EXCEPTION) << "KernelGraph " << kg->ToString() << " has no cnodes!";
  }
  // 4. insert first_label
  auto start_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  for (auto node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimPartial)) {
      InsertControlDependToGraph(kg, NOT_NULL(start_label), NOT_NULL(node));
      break;
    }
  }

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
      RecurseCall(kg, NOT_NULL(cnode), (i + 1 < nodes.size() ? nodes[i + 1] : nullptr), memo);
    } else if (!arg->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Unknown type call node " << cnode->DebugString();
    } else if (IsPrimitiveCNode(arg->cast<CNodePtr>(), prim::kPrimSwitch)) {
      auto arg_cnode = arg->cast<CNodePtr>();
      cnode->set_inputs(cnode->inputs());
      RecurseSwitch(kg, NOT_NULL(cnode), memo);
    } else if (IsPrimitiveCNode(arg->cast<CNodePtr>(), prim::kPrimSwitchLayer)) {
      auto arg_cnode = arg->cast<CNodePtr>();
      cnode->set_inputs(cnode->inputs());
      RecurseSwitchLayer(kg, NOT_NULL(cnode), memo);
    }
  }

  MS_LOG(INFO) << "End KernelGraph process: " << kg->ToString();
  return NOT_NULL(start_label);
}

std::vector<CNodePtr> AscendControlParser::GetCNodes(const std::vector<AnfNodePtr> &in) {
  std::vector<CNodePtr> out;
  for (auto &node : in) {
    if (node->isa<CNode>()) {
      out.push_back(node->cast<CNodePtr>());
    }
  }
  return out;
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
                                          const CNodePtr &last_label, const VectorRef &args) {
  if (from_graph_call_node != nullptr) {
    SetSubGraphInput(kg, NOT_NULL(from_graph_call_node), args);
  }

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
    InsertDependToGraph(kg, NOT_NULL(label_goto));
  }
}

void AscendControlParser::RecurseCall(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node, const CNodePtr &next_node,
                                      NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "process call func " << cur_node->DebugString();

  // 1 get kernel graph
  auto origin_inputs = cur_node->inputs();
  std::vector<AnfNodePtr> new_inputs = {std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelGotoOpName))};
  auto call_args = GetCallArgs(origin_inputs.begin() + 1, origin_inputs.end());
  if (!IsValueNode<KernelGraph>(origin_inputs[kCNodeCallArg])) {
    MS_LOG(WARNING) << "Node " << cur_node->DebugString(10) << " index " << kCNodeCallArg << " is not a ValueNode";
    return;
  }
  // 2 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSetOpName))});
  // 3 add depend relationship
  InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
  if (next_node != nullptr && next_node != kg->get_return()) {
    InsertControlDependToGraph(kg, NOT_NULL(back_label), NOT_NULL(next_node));
  }
  auto call_kg = GetValueNode<KernelGraphPtr>(origin_inputs[kCNodeCallArg]);
  // 4 modify call op to goto op
  cur_node->set_input(kCNodePrim, new_inputs[kCNodePrim]);
  // 5 recurse sub graph
  CNodePtr sub_label = ProcessKernelGraph(NOT_NULL(call_kg), cur_node, back_label, call_args, memo);
  new_inputs.push_back(sub_label);
  new_inputs.insert(new_inputs.end(), origin_inputs.begin(), origin_inputs.end());
  cur_node->set_inputs(new_inputs);
  cur_node->set_abstract(nullptr);
  MS_LOG(INFO) << "success process call func " << cur_node->DebugString();
}

void AscendControlParser::RecurseSwitch(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                                        NotNull<std::set<KernelGraphPtr> *> memo) {
  MS_LOG(INFO) << "process switch node " << cur_node->DebugString();

  if (cur_node->size() < kCNodeSwitchLength) {
    MS_LOG(EXCEPTION) << "Inputs of apply node must more than " << kCNodeSwitchLength;
  }
  // 1 return label
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(prim::kPrimLabelSet)});
  // 2 recurse sub graph
  auto origin_switch_inputs = cur_node->inputs();
  std::vector<AnfNodePtr> new_switch_inputs = {
    std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName)),
    origin_switch_inputs[kCNodeSwitchCond]};
  for (size_t i = kCNodeSwitchCond + 1; i < kCNodeSwitchLength; ++i) {
    // 2.1 branch kernel graph and args
    CNodePtr partial;
    KernelGraphPtr branch_fg;
    VectorRef call_args;
    std::tie(partial, branch_fg, call_args) = ParsePartial(NOT_NULL(origin_switch_inputs[i]));
    // 2.2 add depend relationship
    InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
    // 2.3 recurse sub graph
    CNodePtr branch_label = ProcessKernelGraph(NOT_NULL(branch_fg), cur_node, back_label, call_args, memo);
    new_switch_inputs.push_back(branch_label);
  }
  std::swap(new_switch_inputs[kCNodeSwitchTrue], new_switch_inputs[kCNodeSwitchFalse]);
  new_switch_inputs.insert(new_switch_inputs.end(), origin_switch_inputs.begin(), origin_switch_inputs.end());
  cur_node->set_inputs(new_switch_inputs);
  cur_node->set_abstract(nullptr);
  MS_LOG(INFO) << "success process switch func " << cur_node->DebugString();
}

void AscendControlParser::RecurseSwitchLayer(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> cur_node,
                                             NotNull<std::set<KernelGraphPtr> *> memo) {
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
  auto back_label = kg->NewCNode({std::make_shared<ValueNode>(std::make_shared<Primitive>(kLabelSwitchOpName))});
  // 2 recurse sub graph
  auto origin_switch_inputs = cur_node->inputs();
  std::vector<AnfNodePtr> new_switch_inputs = {std::make_shared<ValueNode>(prim::kPrimLabelSwitch),
                                               origin_switch_inputs[kCNodeSwitchCond]};
  for (size_t i = 0; i < branch_partial.size(); ++i) {
    // 2.1 branch kernel graph and args
    CNodePtr partial;
    KernelGraphPtr branch_fg;
    VectorRef call_args;
    std::tie(partial, branch_fg, call_args) = ParsePartial(NOT_NULL(origin_switch_inputs[i]));
    // 2.2 add depend relationship
    InsertControlDependToGraph(kg, cur_node, NOT_NULL(back_label));
    // 2.3 recurse sub graph
    CNodePtr branch_label = ProcessKernelGraph(NOT_NULL(branch_fg), cur_node, back_label, call_args, memo);
    new_switch_inputs.push_back(branch_label);
  }
  new_switch_inputs.insert(new_switch_inputs.end(), branch_partial.begin(), branch_partial.end());
  cur_node->set_inputs(new_switch_inputs);
  cur_node->set_abstract(nullptr);
  MS_LOG(INFO) << "success process switch layer " << cur_node->DebugString();
}

std::tuple<CNodePtr, KernelGraphPtr, VectorRef> AscendControlParser::ParsePartial(NotNull<AnfNodePtr> node) {
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
  auto call_args = GetCallArgs(partial_inputs.begin() + kCNodePartialFunc + 1, partial_inputs.end());

  return {partial_cnode, branch_kg, call_args};
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

size_t AscendControlParser::SetChildGraphInput(NotNull<KernelGraphPtr> kg, NotNull<AnfNodePtr> node,
                                               size_t input_index) {
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  if (output_num > 1 && !AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    return input_index + output_num;
  }

  auto &graph_inputs = kg->inputs();
  if (input_index >= graph_inputs.size()) {
    MS_LOG(EXCEPTION) << "input_index " << input_index << " out of range size " << graph_inputs.size();
  }
  auto backend_parameter = graph_inputs[input_index];
  if (node.get()->isa<Parameter>()) {
    MS_EXCEPTION_IF_NULL(backend_parameter);
    MS_LOG(INFO) << "Reuse node [" << node->DebugString() << "], old node[" << backend_parameter->DebugString()
                 << "] will be replaced.";
    kg->ReplaceNode(backend_parameter, node);
    return input_index;
  }
  InsertAssignToGraph(kg, node, NOT_NULL(backend_parameter));
  return input_index + 1;
}

void AscendControlParser::SetSubGraphInput(NotNull<KernelGraphPtr> kg, NotNull<CNodePtr> from_graph_call_node,
                                           const VectorRef &args) {}

}  // namespace session
}  // namespace mindspore
