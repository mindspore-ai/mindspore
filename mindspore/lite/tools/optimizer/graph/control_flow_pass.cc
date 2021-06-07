/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/control_flow_pass.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ops/switch.h"
#include "ops/fusion/partial_fusion.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"

namespace mindspore::opt {

ValueNodePtr ControlFlowPass::GetSwitchAnfPrim() {
  auto switch_prim = std::make_shared<mindspore::ops::Switch>();
  ValueNodePtr switch_anf_prim = NewValueNode(switch_prim);
  return switch_anf_prim;
}

ValueNodePtr ControlFlowPass::GetPartialAnfPrim() {
  auto partial_prim = std::make_shared<mindspore::ops::PartialFusion>();
  ValueNodePtr partial_anf_prim = NewValueNode(partial_prim);
  return partial_anf_prim;
}

void ControlFlowPass::ReplaceNode(const FuncGraphPtr &fg,
                                  const std::unordered_map<AnfNodePtr, AnfNodePtr> &replace_pairs) {
  for (auto &node : fg->nodes()) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto new_inputs = cnode->inputs();
    for (auto &input : new_inputs) {
      if (replace_pairs.find(input) == replace_pairs.end()) {
        continue;
      }
      input = replace_pairs.at(input);
    }
    cnode->set_inputs(new_inputs);
  }
}

void ControlFlowPass::FunGraphInputsOnlyUsedByAfterParts(const FuncGraphPtr &fg, const CNodePtr &aim_cnode,
                                                         std::vector<AnfNodePtr> *fg_inputs_only_used_by_after_fg) {
  auto fg_inputs = fg->get_inputs();
  fg_inputs_only_used_by_after_fg->assign(fg_inputs.begin(), fg_inputs.end());
  auto nodes = TopoSort(aim_cnode);
  for (auto it = fg_inputs_only_used_by_after_fg->begin(); it != fg_inputs_only_used_by_after_fg->end();) {
    if (lite::IsContain(nodes, *it)) {
      it = fg_inputs_only_used_by_after_fg->erase(it);
    } else {
      ++it;
    }
  }
}

int ControlFlowPass::SplitGraph(const FuncGraphPtr &fg, const PrimitivePtr &aim_prim, AnfNodePtr *aim_prim_type_node,
                                std::vector<AnfNodePtr> *remain_nodes) {
  auto inputs = fg->get_inputs();
  std::vector<AnfNodePtr> visited_nodes{};
  visited_nodes.assign(inputs.begin(), inputs.end());
  // notice: fg->nodes() is not work in this pass, cause too many useless parameter have been created.
  auto node_list = TopoSort(fg->get_return());
  for (auto &node : node_list) {
    if (utils::isa<CNodePtr>(node) && CheckPrimitiveType(node, aim_prim)) {
      *aim_prim_type_node = node;
      break;
    }
    if (!utils::isa<CNodePtr>(node) && !utils::isa<ParameterPtr>(node)) {
      continue;
    }
    if (!lite::IsContain(visited_nodes, node)) {
      visited_nodes.push_back(node);
    }
  }

  for (auto &node : node_list) {
    if (!lite::IsContain(visited_nodes, node) && node != *aim_prim_type_node) {
      remain_nodes->push_back(node);
    }
  }
  return RET_SUCCESS;
}

int ControlFlowPass::CreateAfterGraph(const FuncGraphPtr &main_fg, const std::vector<AnfNodePtr> &remain_nodes,
                                      const CNodePtr &aim_cnode, FuncGraphPtr *after_fg) {
  *after_fg = std::make_shared<FuncGraph>();
  auto manager = main_fg->manager();
  manager->AddFuncGraph(*after_fg);
  (*after_fg)->set_attr("fmk", MakeValue(static_cast<int>(lite::converter::FmkType_TF)));
  (*after_fg)->set_attr("graph_name", MakeValue(aim_cnode->fullname_with_scope() + "_after_fg"));
  (*after_fg)->set_manager(main_fg->manager());

  for (auto &cur_node : remain_nodes) {
    if (cur_node->isa<ValueNode>()) {
      continue;
    }
    if (cur_node == main_fg->get_return()) {
      continue;
    }
    (*after_fg)->AddNode(cur_node);
    cur_node->set_func_graph(*after_fg);
    if (cur_node == main_fg->output()) {
      (*after_fg)->set_output(cur_node, false);
    }
    main_fg->DropNode(cur_node);
  }
  return RET_SUCCESS;
}

int ControlFlowPass::CreateWhileCondCallNode(
  const FuncGraphPtr &fg, const CNodePtr &while_cnode, std::vector<AnfNodePtr> *fg_inputs_only_used_by_after_fg,
  CNodePtr *cond_call_cnode,
  std::unordered_map<AnfNodePtr, AnfNodePtr> *fg_inputs_and_after_partial_inputs_replace_pairs) {
  auto cond_vnode = while_cnode->input(kWhileCondIndex);
  MS_ASSERT(cond_vnode != nullptr);
  auto cond_fg = GetValueNode<std::shared_ptr<FuncGraph>>(cond_vnode);
  if (cond_fg == nullptr) {
    MS_LOG(ERROR) << "Get value as func graph failed.";
    return RET_FAILED;
  }
  // get fg input which is not used by cond fg
  FunGraphInputsOnlyUsedByAfterParts(fg, while_cnode, fg_inputs_only_used_by_after_fg);

  std::vector<AnfNodePtr> cond_call_cnode_inputs{cond_vnode};
  cond_call_cnode_inputs.insert(cond_call_cnode_inputs.end(), while_cnode->inputs().begin() + kWhileMinInputSize,
                                while_cnode->inputs().end());
  // set after fg inputs to cond_call_cnode inputs
  cond_call_cnode_inputs.insert(cond_call_cnode_inputs.end(), fg_inputs_only_used_by_after_fg->begin(),
                                fg_inputs_only_used_by_after_fg->end());

  *cond_call_cnode = fg->NewCNode(cond_call_cnode_inputs);
  (*cond_call_cnode)->set_fullname_with_scope("CNode_" + cond_fg->get_attr("graph_name")->ToString());

  for (auto &node : *fg_inputs_only_used_by_after_fg) {
    if (!utils::isa<ParameterPtr>(node)) {
      MS_LOG(ERROR) << "fg is not right.";
      return RET_FAILED;
    }
    auto new_parameter = cond_fg->add_parameter();
    new_parameter->set_name(node->fullname_with_scope() + "_cond_fg_parameter");
    new_parameter->set_abstract(node->abstract());
    (*fg_inputs_and_after_partial_inputs_replace_pairs)[node] = new_parameter;
  }

  return RET_SUCCESS;
}

int ControlFlowPass::CreateWhileBodyPartialNode(const FuncGraphPtr &cond_fg, const CNodePtr &while_cnode,
                                                CNodePtr *body_partial_node) {
  auto body_vnode = while_cnode->input(kWhileBodyIndex);
  auto body_fg = GetValueNode<std::shared_ptr<FuncGraph>>(body_vnode);
  if (body_fg == nullptr) {
    MS_LOG(ERROR) << "Get value as func_graph failed.";
    return RET_FAILED;
  }
  if (ProcessWhileOp(body_fg) != RET_SUCCESS) {
    MS_LOG(ERROR) << "ProcessWhileOp failed.";
    return RET_FAILED;
  }
  ValueNodePtr partial_anf_primitive = GetPartialAnfPrim();
  if (partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialAnfPrim failed.";
    return RET_FAILED;
  }

  std::vector<AnfNodePtr> body_partial_node_inputs{partial_anf_primitive, body_vnode};
  // set body inputs to body partial inputs
  auto cond_fg_inputs = cond_fg->get_inputs();
  body_partial_node_inputs.insert(body_partial_node_inputs.end(), cond_fg_inputs.begin(), cond_fg_inputs.end());
  *body_partial_node = cond_fg->NewCNode(body_partial_node_inputs);
  (*body_partial_node)->set_fullname_with_scope("CNode_" + body_fg->get_attr("graph_name")->ToString());

  // add after inputs for body fg to call cond fg
  auto body_fg_inputs = body_fg->get_inputs();
  auto origin_body_fg_inputs_size = body_fg_inputs.size();
  for (size_t i = origin_body_fg_inputs_size; i < cond_fg_inputs.size(); ++i) {
    if (!utils::isa<ParameterPtr>(cond_fg_inputs[i])) {
      MS_LOG(ERROR) << "fg is not right.";
      return RET_FAILED;
    }
    auto cond_fg_input_para = cond_fg_inputs[i]->cast<ParameterPtr>();
    auto new_parameter = body_fg->add_parameter();
    new_parameter->set_name(cond_fg_inputs[i]->fullname_with_scope() + "_body_fg_parameter");
    new_parameter->set_abstract(cond_fg_inputs[i]->abstract());
  }

  // call the cond fg
  auto cond_partial_vnode = NewValueNode(cond_fg);
  std::vector<AnfNodePtr> cond_call_cnode_inputs{cond_partial_vnode};
  // set body fg output
  auto body_output = body_fg->output()->cast<CNodePtr>();
  MS_ASSERT(body_output != nullptr);
  if (CheckPrimitiveType(body_output, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < body_output->inputs().size(); ++i) {
      cond_call_cnode_inputs.push_back(body_output->input(i));
    }
    body_fg->DropNode(body_output);
  } else {
    cond_call_cnode_inputs.push_back(body_output);
  }

  body_fg_inputs = body_fg->get_inputs();
  for (size_t i = origin_body_fg_inputs_size; i < body_fg_inputs.size(); ++i) {
    cond_call_cnode_inputs.push_back(body_fg_inputs[i]);
  }

  auto cond_call_cnode = body_fg->NewCNode(cond_call_cnode_inputs);
  cond_call_cnode->set_fullname_with_scope(body_fg->get_attr("graph_name")->ToString() + "_call_cond_fg");
  body_fg->set_output(cond_call_cnode);
  return RET_SUCCESS;
}

int ControlFlowPass::CreateWhileAfterPartialNode(
  const FuncGraphPtr &main_fg, const FuncGraphPtr &cond_fg, const std::vector<AnfNodePtr> &remain_nodes,
  const std::vector<AnfNodePtr> &fg_inputs_only_used_by_after_fg,
  const std::unordered_map<AnfNodePtr, AnfNodePtr> &fg_inputs_and_after_partial_inputs_replace_pairs,
  CNodePtr *while_cnode, CNodePtr *after_partial_cnode) {
  // create after_fg
  FuncGraphPtr after_fg = nullptr;
  if (CreateAfterGraph(main_fg, remain_nodes, *while_cnode, &after_fg) != RET_SUCCESS) {
    MS_LOG(ERROR) << "CreateAfterGraph failed.";
    return RET_FAILED;
  }

  auto after_value_node = NewValueNode(after_fg);
  ValueNodePtr partial_anf_primitive = GetPartialAnfPrim();
  if (partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialAnfPrim failed.";
    return RET_FAILED;
  }

  std::unordered_map<AnfNodePtr, AnfNodePtr> while_output_replace_pairs{};
  std::vector<AnfNodePtr> after_partial_cnode_inputs{partial_anf_primitive, after_value_node};
  auto cond_fg_inputs = cond_fg->get_inputs();
  for (const auto &node : after_fg->nodes()) {
    if (!CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto get_tuple_item_cnode = node->cast<CNodePtr>();
    MS_ASSERT(get_tuple_item_cnode->inputs().size() == kGetItemInputSize);
    if (get_tuple_item_cnode->input(kCNodeFirstInputIndex) != *while_cnode) {
      continue;
    }
    auto index_vnode = get_tuple_item_cnode->inputs().at(kCNodeSecondInputIndex);
    if (!utils::isa<ValueNode>(index_vnode)) {
      MS_LOG(ERROR) << "TupleGetItem's input 2 is not value node";
      return RET_FAILED;
    }
    auto value_node = utils::cast<ValueNodePtr>(index_vnode);
    MS_ASSERT(value_node != nullptr);

    auto input_index = value_node->value()->type()->number_type() == kNumberTypeInt64
                         ? GetValue<int64_t>(value_node->value())
                         : GetValue<int>(value_node->value());

    after_partial_cnode_inputs.push_back(cond_fg_inputs.at(input_index));
    auto new_parameter = after_fg->add_parameter();
    new_parameter->set_name(node->fullname_with_scope() + "_after_partial_parameter");
    new_parameter->set_abstract(node->abstract());
    while_output_replace_pairs[node] = new_parameter;
  }

  for (auto &pair : while_output_replace_pairs) {
    // get all nodes in after_fg
    after_fg->manager()->Replace(pair.first, pair.second);
    after_fg->DropNode(pair.first);
  }

  std::unordered_map<AnfNodePtr, AnfNodePtr> after_partial_replace_pairs{};
  for (auto &input : fg_inputs_only_used_by_after_fg) {
    after_partial_cnode_inputs.push_back(fg_inputs_and_after_partial_inputs_replace_pairs.at(input));
    auto new_parameter = after_fg->add_parameter();
    new_parameter->set_name(input->fullname_with_scope() + "_after_fg_parameter");
    new_parameter->set_abstract(input->abstract());
    after_partial_replace_pairs[input] = new_parameter;
  }

  ReplaceNode(after_fg, after_partial_replace_pairs);
  *after_partial_cnode = cond_fg->NewCNode(after_partial_cnode_inputs);
  (*after_partial_cnode)->set_fullname_with_scope("CNode_" + after_fg->get_attr("graph_name")->ToString());
  return RET_SUCCESS;
}

int ControlFlowPass::ProcessWhileOp(const FuncGraphPtr &fg) {
  if (fg == nullptr) {
    MS_LOG(ERROR) << "fg is nullptr.";
    return RET_FAILED;
  }

  AnfNodePtr while_node = nullptr;
  std::vector<AnfNodePtr> remain_nodes{};
  int ret = SplitGraph(fg, prim::kPrimWhile, &while_node, &remain_nodes);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "SplitGraph failed, ret: " << ret;
    return ret;
  }

  if (while_node == nullptr) {
    MS_LOG(INFO) << "not found while, not need to process.";
    return RET_SUCCESS;
  }

  auto while_cnode = while_node->cast<CNodePtr>();
  MS_ASSERT(while_cnode != nullptr);
  if (while_cnode->inputs().size() < kWhileMinInputSize) {
    MS_LOG(ERROR) << "while input is not right.";
    return RET_FAILED;
  }

  CNodePtr cond_call_cnode = nullptr;
  std::vector<AnfNodePtr> fg_inputs_only_used_by_after_fg{};
  std::unordered_map<AnfNodePtr, AnfNodePtr> fg_inputs_and_after_partial_inputs_replace_pairs{};
  ret = CreateWhileCondCallNode(fg, while_cnode, &fg_inputs_only_used_by_after_fg, &cond_call_cnode,
                                &fg_inputs_and_after_partial_inputs_replace_pairs);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "while create cond call cnode failed, ret: " << ret;
    return ret;
  }

  AnfNodePtr cond_fg_vnode = cond_call_cnode->input(kCNodePrimIndex);
  MS_ASSERT(cond_fg_vnode != nullptr);
  auto cond_fg = GetValueNode<std::shared_ptr<FuncGraph>>(cond_fg_vnode);
  if (cond_fg == nullptr) {
    MS_LOG(ERROR) << "Get value as func_graph failed.";
    return RET_FAILED;
  }

  CNodePtr body_partial_node = nullptr;
  ret = CreateWhileBodyPartialNode(cond_fg, while_cnode, &body_partial_node);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "while create body partial cnode failed, ret: " << ret;
    return ret;
  }

  CNodePtr after_partial_cnode = nullptr;
  ret =
    CreateWhileAfterPartialNode(fg, cond_fg, remain_nodes, fg_inputs_only_used_by_after_fg,
                                fg_inputs_and_after_partial_inputs_replace_pairs, &while_cnode, &after_partial_cnode);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "while create after partial cnode failed, ret: " << ret;
    return ret;
  }

  // create switch cnode
  ValueNodePtr switch_anf_primitive = GetSwitchAnfPrim();
  if (switch_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetSwitchAnfPrim failed.";
    return false;
  }

  // insert switch node
  std::vector<AnfNodePtr> switch_node_inputs = {switch_anf_primitive, cond_fg->output(), body_partial_node,
                                                after_partial_cnode};
  auto switch_cnode = cond_fg->NewCNode(switch_node_inputs);
  switch_cnode->set_fullname_with_scope("Switch-" + cond_fg->get_attr("graph_name")->ToString());

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{switch_cnode};
  auto call_node = cond_fg->NewCNode(call_node_inputs);
  call_node->set_fullname_with_scope("call_" + switch_cnode->fullname_with_scope());
  cond_fg->set_output(call_node);
  fg->DropNode(while_cnode);
  fg->set_output(cond_call_cnode);

  FuncGraphPtr after_fg =
    after_partial_cnode->input(kCNodeFirstInputIndex)->cast<ValueNodePtr>()->value()->cast<FuncGraphPtr>();
  if (after_fg == nullptr) {
    MS_LOG(ERROR) << "after_fg is nullptr.";
    return RET_FAILED;
  }

  if (!Run(after_fg)) {
    MS_LOG(ERROR) << "process control flow for after fg failed.";
    return RET_FAILED;
  }
  return RET_SUCCESS;
}

int ControlFlowPass::CreateIfPartialNode(const FuncGraphPtr &fg,
                                         const std::vector<AnfNodePtr> &fg_inputs_only_used_by_after_partial,
                                         const size_t &index, CNodePtr *if_cnode, FuncGraphPtr *after_fg,
                                         CNodePtr *then_partial_cnode) {
  auto then_vnode = (*if_cnode)->input(index);
  MS_ASSERT(then_vnode != nullptr);
  auto then_fg = GetValueNode<std::shared_ptr<FuncGraph>>(then_vnode);
  if (then_fg == nullptr) {
    MS_LOG(ERROR) << "Get value as func_graph failed.";
    return RET_FAILED;
  }

  // create after partial node
  ValueNodePtr then_partial_anf_primitive = GetPartialAnfPrim();
  if (then_partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialAnfPrim failed.";
    return RET_FAILED;
  }
  std::vector<AnfNodePtr> then_partial_cnode_inputs{then_partial_anf_primitive, then_vnode};
  then_partial_cnode_inputs.insert(then_partial_cnode_inputs.end(), (*if_cnode)->inputs().begin() + kIfMinInputSize,
                                   (*if_cnode)->inputs().end());

  // set fg inputs to then_partial_cnode inputs
  then_partial_cnode_inputs.insert(then_partial_cnode_inputs.end(), fg_inputs_only_used_by_after_partial.begin(),
                                   fg_inputs_only_used_by_after_partial.end());

  *then_partial_cnode = fg->NewCNode(then_partial_cnode_inputs);
  auto then_fg_name = then_fg->get_attr("graph_name")->ToString();
  (*then_partial_cnode)->set_fullname_with_scope("partial_" + then_fg_name);

  std::unordered_map<AnfNodePtr, AnfNodePtr> then_fg_inputs_and_fg_inputs_replace_pairs{};
  std::vector<AnfNodePtr> new_parameters{};
  for (auto &node : fg_inputs_only_used_by_after_partial) {
    if (!utils::isa<ParameterPtr>(node)) {
      MS_LOG(ERROR) << "fg is not right.";
      return RET_FAILED;
    }
    auto new_parameter = then_fg->add_parameter();
    new_parameter->set_name(node->fullname_with_scope() + "_" + then_fg_name + "_parameter");
    new_parameter->set_abstract(node->abstract());
    then_fg_inputs_and_fg_inputs_replace_pairs[node] = new_parameter;
    new_parameters.push_back(new_parameter);
  }

  // create after partial node
  ValueNodePtr after_partial_anf_primitive = GetPartialAnfPrim();
  if (after_partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialAnfPrim failed.";
    return RET_FAILED;
  }
  auto after_value_node = NewValueNode(*after_fg);
  // make the right after partial input
  std::vector<AnfNodePtr> after_partial_cnode_inputs{after_partial_anf_primitive, after_value_node};
  auto then_fg_output = then_fg->output()->cast<CNodePtr>();
  if (!CheckPrimitiveType(then_fg_output, prim::kPrimMakeTuple)) {
    after_partial_cnode_inputs.push_back(then_fg_output);
  } else {
    for (size_t i = kCNodeFirstInputIndex; i < then_fg_output->inputs().size(); ++i) {
      after_partial_cnode_inputs.push_back(then_fg_output->input(i));
    }
  }

  // add after fg inputs to partial node
  std::copy(new_parameters.begin(), new_parameters.end(), std::back_inserter(after_partial_cnode_inputs));

  // insert partial node
  auto after_partial_cnode = then_fg->NewCNode(after_partial_cnode_inputs);
  auto after_fg_name = (*after_fg)->get_attr("graph_name")->ToString();
  after_partial_cnode->set_fullname_with_scope("partial_" + after_fg_name);

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{after_partial_cnode};
  auto call_node = then_fg->NewCNode(call_node_inputs);
  call_node->set_fullname_with_scope("call_" + after_partial_cnode->fullname_with_scope());
  then_fg->set_output(call_node);
  then_fg->DropNode(then_fg_output);

  // check the inputs of after fg
  auto after_fg_inputs_size = (*after_fg)->get_inputs().size();
  if (after_fg_inputs_size == after_partial_cnode_inputs.size() - 2) {
    MS_LOG(INFO) << "not need add after fg input parameters.";
    return RET_SUCCESS;
  }
  // make the inputs of the after fg
  std::unordered_map<AnfNodePtr, AnfNodePtr> after_partial_after_fg_replace_pairs{};
  std::unordered_map<AnfNodePtr, AnfNodePtr> if_cnode_after_fg_replace_pairs{};
  for (size_t i = kPartialFirstInputSize; i < after_partial_cnode_inputs.size(); ++i) {
    auto &input = after_partial_cnode_inputs[i];
    auto new_parameter = (*after_fg)->add_parameter();
    new_parameter->set_name(input->fullname_with_scope() + "_after_fg_parameter");
    new_parameter->set_abstract(input->abstract());
    after_partial_after_fg_replace_pairs[input] = new_parameter;
    if (i < kPartialFirstInputSize + (*if_cnode)->size() - kIfMinInputSize) {
      after_partial_after_fg_replace_pairs[*if_cnode] = new_parameter;
    }
  }
  ReplaceNode(*after_fg, then_fg_inputs_and_fg_inputs_replace_pairs);
  ReplaceNode(*after_fg, after_partial_after_fg_replace_pairs);
  return RET_SUCCESS;
}

int ControlFlowPass::CreateIfElsePartialNode(const FuncGraphPtr &main_fg,
                                             const std::vector<AnfNodePtr> &fg_inputs_only_used_by_after_partial,
                                             CNodePtr *if_cnode, FuncGraphPtr *after_fg, CNodePtr *else_partial_cnode) {
  return CreateIfPartialNode(main_fg, fg_inputs_only_used_by_after_partial, kIfElseIndex, if_cnode, after_fg,
                             else_partial_cnode);
}

int ControlFlowPass::CreateIfThenPartialNode(const FuncGraphPtr &main_fg,
                                             const std::vector<AnfNodePtr> &fg_inputs_only_used_by_after_partial,
                                             CNodePtr *if_cnode, FuncGraphPtr *after_fg, CNodePtr *then_partial_cnode) {
  return CreateIfPartialNode(main_fg, fg_inputs_only_used_by_after_partial, kIfThenIndex, if_cnode, after_fg,
                             then_partial_cnode);
}

int ControlFlowPass::ProcessIfOp(const FuncGraphPtr &fg) {
  if (fg == nullptr) {
    MS_LOG(ERROR) << "fg is nullptr.";
    return RET_FAILED;
  }
  AnfNodePtr if_node = nullptr;
  std::vector<AnfNodePtr> remain_nodes{};
  int ret = SplitGraph(fg, prim::kPrimIf, &if_node, &remain_nodes);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "SplitGraph failed, ret: " << ret;
    return ret;
  }

  if (if_node == nullptr) {
    MS_LOG(INFO) << "not found if, not need to process.";
    return RET_SUCCESS;
  }

  auto if_cnode = if_node->cast<CNodePtr>();
  MS_ASSERT(if_cnode != nullptr);
  if (if_cnode->inputs().size() < kIfMinInputSize) {
    MS_LOG(ERROR) << "if input is not right.";
    return RET_FAILED;
  }

  // create after_fg
  FuncGraphPtr after_fg = nullptr;
  if (CreateAfterGraph(fg, remain_nodes, if_cnode, &after_fg) != RET_SUCCESS) {
    MS_LOG(ERROR) << "CreateAfterGraph failed.";
    return RET_FAILED;
  }

  // get fg input which is not used by after_parts
  std::vector<AnfNodePtr> fg_inputs_only_used_by_after_partial{};
  FunGraphInputsOnlyUsedByAfterParts(fg, if_cnode, &fg_inputs_only_used_by_after_partial);

  CNodePtr then_partial_cnode = nullptr;
  ret = CreateIfThenPartialNode(fg, fg_inputs_only_used_by_after_partial, &if_cnode, &after_fg, &then_partial_cnode);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "if create then partial cnode failed, ret: " << ret;
    return ret;
  }

  CNodePtr else_partial_cnode = nullptr;
  ret = CreateIfElsePartialNode(fg, fg_inputs_only_used_by_after_partial, &if_cnode, &after_fg, &else_partial_cnode);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "if create else partial cnode failed, ret: " << ret;
    return ret;
  }

  // create switch cnode
  ValueNodePtr switch_anf_primitive = GetSwitchAnfPrim();
  if (switch_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetSwitchAnfPrim failed.";
    return false;
  }

  //  insert switch node
  std::vector<AnfNodePtr> switch_node_inputs = {switch_anf_primitive, if_cnode->input(kIfCondIndex), then_partial_cnode,
                                                else_partial_cnode};
  auto switch_cnode = fg->NewCNode(switch_node_inputs);
  switch_cnode->set_fullname_with_scope("Switch-" + fg->get_attr("graph_name")->ToString());

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{switch_cnode};
  auto call_node = fg->NewCNode(call_node_inputs);
  call_node->set_fullname_with_scope("call_" + switch_cnode->fullname_with_scope());
  fg->DropNode(if_cnode);
  fg->set_output(call_node);

  if (!Run(after_fg)) {
    MS_LOG(ERROR) << "process control flow for after fg failed.";
    return RET_FAILED;
  }

  return RET_SUCCESS;
}

bool ControlFlowPass::Run(const FuncGraphPtr &fg) {
  int ret = ProcessWhileOp(fg);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "ProcessWhileOp failed.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return false;
  }
  ret = ProcessIfOp(fg);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "ProcessIfOp failed.";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return false;
  }
  return true;
}
}  // namespace mindspore::opt
