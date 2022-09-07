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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/control_flow_pass.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ops/switch.h"
#include "ops/fusion/partial_fusion.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"
#include "tools/common/node_util.h"
#include "nnacl/op_base.h"
#include "include/registry/converter_context.h"

namespace mindspore::opt {
void ControlFlowPass::ReplaceNode(const FuncGraphPtr &fg,
                                  const std::unordered_map<AnfNodePtr, AnfNodePtr> &replace_pairs) {
  for (auto &node : fg->nodes()) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
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

void ControlFlowPass::VisitedNodesUsedByAfterParts(const std::set<AnfNodePtr> &visited_nodes,
                                                   const std::vector<AnfNodePtr> &remain_nodes,
                                                   std::vector<AnfNodePtr> *visited_nodes_used_by_after_fg) {
  std::deque<AnfNodePtr> nodes{};
  std::set<AnfNodePtr> visited_nodes_used_by_after_fg_set{};
  std::set<AnfNodePtr> remain_nodes_set{};
  nodes.assign(remain_nodes.begin(), remain_nodes.end());
  while (!nodes.empty()) {
    auto node = nodes.front();
    nodes.pop_front();
    remain_nodes_set.insert(node);
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    for (auto &input : cnode->inputs()) {
      if (visited_nodes.find(input) != visited_nodes.end() &&
          visited_nodes_used_by_after_fg_set.find(input) == visited_nodes_used_by_after_fg_set.end()) {
        visited_nodes_used_by_after_fg->push_back(input);
        visited_nodes_used_by_after_fg_set.insert(input);
      }
    }
  }
}

size_t ControlFlowPass::GetItemVisitedNums(const std::set<AnfNodePtr> &visited_nodes, const AnfNodePtr &tuple_node) {
  size_t count = 0;
  for (auto &node : visited_nodes) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto get_item_cnode = node->cast<CNodePtr>();
    MS_ASSERT(get_item_cnode != nullptr);
    if (get_item_cnode->inputs()[kCNodeFirstInputIndex] == tuple_node) {
      count++;
    }
  }
  return count;
}

void ControlFlowPass::MoveGetItemToVisited(const size_t &need_size, const AnfNodePtr &tuple_node,
                                           std::set<AnfNodePtr> *visited_nodes, std::vector<AnfNodePtr> *remain_nodes) {
  size_t i = 0;
  for (auto it = remain_nodes->begin(); it != remain_nodes->end();) {
    if (!utils::isa<CNodePtr>(*it)) {
      ++it;
      continue;
    }
    if (!CheckPrimitiveType(*it, prim::kPrimTupleGetItem)) {
      ++it;
      continue;
    }
    auto get_item_cnode = (*it)->cast<CNodePtr>();
    MS_ASSERT(get_item_cnode != nullptr);
    if (get_item_cnode->inputs()[kCNodeFirstInputIndex] != tuple_node) {
      ++it;
      continue;
    }
    i++;
    visited_nodes->insert(*it);
    it = remain_nodes->erase(it);
    if (need_size == i) {
      return;
    }
  }
  MS_LOG(INFO) << tuple_node->fullname_with_scope() << " not found enough get item, size: " << need_size - i;
}

void ControlFlowPass::BindGetItemNodes(std::set<AnfNodePtr> *visited_nodes, std::vector<AnfNodePtr> *remain_nodes) {
  std::deque<AnfNodePtr> multi_output_nodes{};
  for (auto &node : *visited_nodes) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (utils::isa<abstract::AbstractTuple>(node->abstract())) {
      multi_output_nodes.push_back(node);
    }
  }

  while (!multi_output_nodes.empty()) {
    auto cur_node = multi_output_nodes.front();
    multi_output_nodes.pop_front();
    size_t total_getitem_size = cur_node->abstract()->cast<abstract::AbstractTuplePtr>()->size();
    size_t visited_getitem_size = GetItemVisitedNums(*visited_nodes, cur_node);
    if (total_getitem_size == visited_getitem_size) {
      continue;
    }

    size_t need_getitem_size = total_getitem_size - visited_getitem_size;
    MoveGetItemToVisited(need_getitem_size, cur_node, visited_nodes, remain_nodes);
  }
}

int ControlFlowPass::SplitGraph(const FuncGraphPtr &fg, AnfNodePtr *control_flow_node,
                                std::set<AnfNodePtr> *visited_nodes, std::vector<AnfNodePtr> *remain_nodes) {
  auto inputs = fg->get_inputs();

  // notice: fg->nodes() is not work in this pass, cause too many useless parameter have been created.
  auto node_list = TopoSort(fg->get_return());
  for (auto &node : node_list) {
    MS_ASSERT(node != nullptr);
    if (utils::isa<CNodePtr>(node) &&
        (CheckPrimitiveType(node, prim::kPrimWhile) || CheckPrimitiveType(node, prim::kPrimIf))) {
      *control_flow_node = node;
      break;
    }
  }

  std::deque<AnfNodePtr> q;
  visited_nodes->insert(inputs.begin(), inputs.end());
  q.push_back(*control_flow_node);
  while (!q.empty()) {
    auto node = q.front();
    q.pop_front();
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    visited_nodes->insert(node);
    auto cnode = utils::cast<CNodePtr>(node);
    MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cast ptr failed");
    for (size_t i = 0; i < cnode->inputs().size(); i++) {
      auto input = cnode->input(i);
      if (visited_nodes->find(input) == visited_nodes->end()) {
        q.push_back(input);
      }
    }
  }

  for (auto &node : node_list) {
    if (visited_nodes->find(node) == visited_nodes->end()) {
      remain_nodes->push_back(node);
    }
  }
  visited_nodes->erase(*control_flow_node);

  BindGetItemNodes(visited_nodes, remain_nodes);

  return RET_SUCCESS;
}

int ControlFlowPass::CreateAfterGraph(const FuncGraphPtr &main_fg, const std::vector<AnfNodePtr> &remain_nodes,
                                      const CNodePtr &aim_cnode, FuncGraphPtr *after_fg) {
  *after_fg = std::make_shared<FuncGraph>();
  MS_CHECK_TRUE_MSG(*after_fg != nullptr, lite::RET_NULL_PTR, "*after_fg is nullptr");
  auto manager = main_fg->manager();
  MS_ASSERT(manager != nullptr);
  manager->AddFuncGraph(*after_fg);
  (*after_fg)->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeTf)));
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
    if (!utils::isa<ValueNodePtr>(cur_node)) {
      cur_node->set_func_graph(*after_fg);
    }
    if (cur_node == main_fg->output()) {
      (*after_fg)->set_output(cur_node, false);
    }
    main_fg->DropNode(cur_node);
  }
  return RET_SUCCESS;
}

int ControlFlowPass::CreateWhileCondCallNode(
  const FuncGraphPtr &fg, const CNodePtr &while_cnode, const std::vector<AnfNodePtr> &visited_nodes_used_by_after_fg,
  CNodePtr *cond_call_cnode, std::vector<AnfNodePtr> *cond_nodes_used_by_after_partial,
  std::unordered_map<AnfNodePtr, AnfNodePtr> *visited_nodes_and_cond_fg_inputs_replace_pairs) {
  auto cond_vnode = while_cnode->input(kWhileCondIndex);
  MS_CHECK_TRUE_MSG(cond_vnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
  auto cond_fg = GetValueNode<std::shared_ptr<FuncGraph>>(cond_vnode);
  if (cond_fg == nullptr) {
    MS_LOG(ERROR) << "Get value as func graph failed.";
    return RET_FAILED;
  }

  // create after partial node
  ValueNodePtr cond_partial_anf_primitive = lite::GetPartialFusionPrim();
  if (cond_partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialFusionPrim failed.";
    return RET_FAILED;
  }

  std::vector<AnfNodePtr> cond_partial_cnode_inputs{cond_partial_anf_primitive, cond_vnode};
  cond_partial_cnode_inputs.insert(cond_partial_cnode_inputs.end(), while_cnode->inputs().begin() + kWhileMinInputSize,
                                   while_cnode->inputs().end());

  auto origin_cond_fg_inputs = cond_fg->get_inputs();
  for (auto &item : visited_nodes_used_by_after_fg) {
    bool found = false;
    size_t input_index = 0;
    for (size_t i = kPartialFirstInputSize; i < cond_partial_cnode_inputs.size(); ++i) {
      if (cond_partial_cnode_inputs[i] == item) {
        found = true;
        input_index = i - kPartialFirstInputSize;
        break;
      }
    }

    if (found) {
      (*visited_nodes_and_cond_fg_inputs_replace_pairs)[item] = origin_cond_fg_inputs.at(input_index);
      cond_nodes_used_by_after_partial->push_back(origin_cond_fg_inputs.at(input_index));
      continue;
    }

    // set after fg inputs to cond_partial_cnode inputs
    cond_partial_cnode_inputs.push_back(item);
    auto new_parameter = cond_fg->add_parameter();
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, lite::RET_NULL_PTR, "new_parameter is nullptr");
    new_parameter->set_name(item->fullname_with_scope() + "_cond_fg_parameter");
    new_parameter->set_abstract(item->abstract());
    (*visited_nodes_and_cond_fg_inputs_replace_pairs)[item] = new_parameter;
    cond_nodes_used_by_after_partial->push_back(new_parameter);
  }

  auto cond_partial_cnode = fg->NewCNode(cond_partial_cnode_inputs);
  MS_CHECK_TRUE_MSG(cond_partial_cnode != nullptr, lite::RET_NULL_PTR, "cond_partial_cnode is nullptr");
  cond_partial_cnode->set_fullname_with_scope("partial_" + cond_fg->get_attr("graph_name")->ToString());

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{cond_partial_cnode};
  *cond_call_cnode = fg->NewCNode(call_node_inputs);
  MS_CHECK_TRUE_MSG(*cond_call_cnode != nullptr, lite::RET_NULL_PTR, "new cnode is nullptr");
  (*cond_call_cnode)->set_fullname_with_scope("call_" + cond_partial_cnode->fullname_with_scope());

  return RET_SUCCESS;
}

int ControlFlowPass::CreateWhileBodyPartialNode(const FuncGraphPtr &cond_fg, const CNodePtr &while_cnode,
                                                CNodePtr *body_partial_node) {
  auto body_vnode = while_cnode->input(kWhileBodyIndex);
  MS_CHECK_TRUE_MSG(body_vnode != nullptr, RET_FAILED, "body_vnode is nullptr");
  auto body_fg = GetValueNode<std::shared_ptr<FuncGraph>>(body_vnode);
  if (body_fg == nullptr) {
    MS_LOG(ERROR) << "Get value as func_graph failed.";
    return RET_FAILED;
  }

  ValueNodePtr partial_anf_primitive = lite::GetPartialFusionPrim();
  if (partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialFusionPrim failed.";
    return RET_FAILED;
  }

  std::vector<AnfNodePtr> body_partial_node_inputs{partial_anf_primitive, body_vnode};
  // set body inputs to body partial inputs
  auto cond_fg_inputs = cond_fg->get_inputs();
  body_partial_node_inputs.insert(body_partial_node_inputs.end(), cond_fg_inputs.begin(), cond_fg_inputs.end());
  *body_partial_node = cond_fg->NewCNode(body_partial_node_inputs);
  MS_CHECK_TRUE_MSG(*body_partial_node != nullptr, RET_FAILED, "new cnode is nullptr");
  (*body_partial_node)->set_fullname_with_scope("CNode_" + body_fg->get_attr("graph_name")->ToString());

  // add after inputs for body fg to call cond fg
  auto body_fg_inputs = body_fg->get_inputs();
  auto origin_body_fg_inputs_size = body_fg_inputs.size();
  for (size_t i = origin_body_fg_inputs_size; i < cond_fg_inputs.size(); ++i) {
    if (!utils::isa<ParameterPtr>(cond_fg_inputs[i])) {
      MS_LOG(ERROR) << "fg is not right.";
      return RET_FAILED;
    }
    auto new_parameter = body_fg->add_parameter();
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, lite::RET_NULL_PTR, "new_parameter is nullptr");
    new_parameter->set_name(cond_fg_inputs[i]->fullname_with_scope() + "_body_fg_parameter");
    new_parameter->set_abstract(cond_fg_inputs[i]->abstract());
  }

  // call the cond fg
  ValueNodePtr cond_partial_anf_primitive = lite::GetPartialFusionPrim();
  if (cond_partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "`new cond_partial_anf_primitive failed.";
    return RET_FAILED;
  }
  auto cond_partial_vnode = NewValueNode(cond_fg);
  MS_CHECK_TRUE_MSG(cond_partial_vnode != nullptr, lite::RET_NULL_PTR, "cond_partial_vnode is nullptr");
  std::vector<AnfNodePtr> cond_partial_inputs{cond_partial_anf_primitive, cond_partial_vnode};
  // set body fg output
  auto body_output = body_fg->output()->cast<CNodePtr>();
  MS_ASSERT(body_output != nullptr);
  if (CheckPrimitiveType(body_output, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < body_output->inputs().size(); ++i) {
      cond_partial_inputs.push_back(body_output->input(i));
    }
    body_fg->DropNode(body_output);
  } else {
    cond_partial_inputs.push_back(body_output);
  }

  body_fg_inputs = body_fg->get_inputs();
  for (size_t i = origin_body_fg_inputs_size; i < body_fg_inputs.size(); ++i) {
    cond_partial_inputs.push_back(body_fg_inputs[i]);
  }

  auto cond_partial_cnode = body_fg->NewCNode(cond_partial_inputs);
  MS_CHECK_TRUE_MSG(cond_partial_cnode != nullptr, lite::RET_NULL_PTR, "cond_partial_cnode != nullptr");
  cond_partial_cnode->set_fullname_with_scope(body_fg->get_attr("graph_name")->ToString() + "_call_cond_fg");

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{cond_partial_cnode};
  auto cond_call_cnode = body_fg->NewCNode(call_node_inputs);
  MS_CHECK_TRUE_MSG(cond_call_cnode != nullptr, RET_FAILED, "new cnode is nullptr");
  cond_call_cnode->set_fullname_with_scope("call_" + cond_partial_cnode->fullname_with_scope());
  body_fg->set_output(cond_call_cnode);

  to_process_q.push_back(body_fg);
  return RET_SUCCESS;
}

int ControlFlowPass::CreateWhileAfterPartialNode(
  const FuncGraphPtr &main_fg, const FuncGraphPtr &cond_fg, const std::vector<AnfNodePtr> &remain_nodes,
  const std::vector<AnfNodePtr> &cond_nodes_used_by_after_partial,
  const std::unordered_map<AnfNodePtr, AnfNodePtr> &visited_nodes_and_cond_fg_inputs_replace_pairs,
  const CNodePtr *while_cnode, CNodePtr *after_partial_cnode) {
  // create after_fg
  FuncGraphPtr after_fg = nullptr;
  if (CreateAfterGraph(main_fg, remain_nodes, *while_cnode, &after_fg) != RET_SUCCESS) {
    MS_LOG(ERROR) << "CreateAfterGraph failed.";
    return RET_FAILED;
  }

  auto after_value_node = NewValueNode(after_fg);
  MS_CHECK_TRUE_MSG(after_value_node != nullptr, RET_FAILED, "after_value_node is nullptr");
  ValueNodePtr partial_anf_primitive = lite::GetPartialFusionPrim();
  if (partial_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetPartialFusionPrim failed.";
    return RET_FAILED;
  }

  std::unordered_map<AnfNodePtr, AnfNodePtr> after_partial_inputs_and_after_fg_inputs_replace_pairs{};
  std::vector<AnfNodePtr> after_partial_cnode_inputs{partial_anf_primitive, after_value_node};
  auto cond_fg_inputs = cond_fg->get_inputs();
  for (const auto &node : after_fg->nodes()) {
    if (!CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto get_tuple_item_cnode = node->cast<CNodePtr>();
    MS_ASSERT(get_tuple_item_cnode != nullptr);
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
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, RET_FAILED, "new_parameter != nullptr");
    new_parameter->set_name(node->fullname_with_scope() + "_after_partial_parameter");
    new_parameter->set_abstract(node->abstract());
    after_partial_inputs_and_after_fg_inputs_replace_pairs[node] = new_parameter;
  }

  std::unordered_map<AnfNodePtr, AnfNodePtr> visited_nodes_after_fg_replace_pair{};
  for (auto &input : cond_nodes_used_by_after_partial) {
    after_partial_cnode_inputs.push_back(visited_nodes_and_cond_fg_inputs_replace_pairs.at(input));
    auto new_parameter = after_fg->add_parameter();
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, RET_FAILED, "new_parameter != nullptr");
    new_parameter->set_name(input->fullname_with_scope() + "_after_fg_parameter");
    new_parameter->set_abstract(input->abstract());
    visited_nodes_after_fg_replace_pair[visited_nodes_and_cond_fg_inputs_replace_pairs.at(input)] = new_parameter;
  }

  ReplaceNode(after_fg, visited_nodes_and_cond_fg_inputs_replace_pairs);
  ReplaceNode(after_fg, after_partial_inputs_and_after_fg_inputs_replace_pairs);
  ReplaceNode(after_fg, visited_nodes_after_fg_replace_pair);
  *after_partial_cnode = cond_fg->NewCNode(after_partial_cnode_inputs);
  MS_CHECK_TRUE_MSG(*after_partial_cnode != nullptr, RET_FAILED, "new cnode is nullptr");
  (*after_partial_cnode)->set_fullname_with_scope("CNode_" + after_fg->get_attr("graph_name")->ToString());
  return RET_SUCCESS;
}

int ControlFlowPass::ProcessWhileOp(const FuncGraphPtr &fg, const std::set<AnfNodePtr> &visited_nodes,
                                    const std::vector<AnfNodePtr> &remain_nodes, const AnfNodePtr &while_node) {
  if (while_node == nullptr) {
    MS_LOG(INFO) << "not found while, no need to process.";
    return RET_SUCCESS;
  }

  auto while_cnode = while_node->cast<CNodePtr>();
  MS_ASSERT(while_cnode != nullptr);
  if (while_cnode->inputs().size() < kWhileMinInputSize) {
    MS_LOG(ERROR) << "while input is not right.";
    return RET_FAILED;
  }

  std::vector<AnfNodePtr> visited_nodes_used_by_after_fg{};
  VisitedNodesUsedByAfterParts(visited_nodes, remain_nodes, &visited_nodes_used_by_after_fg);

  CNodePtr cond_call_cnode = nullptr;
  std::unordered_map<AnfNodePtr, AnfNodePtr> visited_nodes_and_cond_fg_inputs_replace_pairs{};
  std::vector<AnfNodePtr> cond_nodes_used_by_after_partial{};
  int ret = CreateWhileCondCallNode(fg, while_cnode, visited_nodes_used_by_after_fg, &cond_call_cnode,
                                    &cond_nodes_used_by_after_partial, &visited_nodes_and_cond_fg_inputs_replace_pairs);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "while create cond call cnode failed, ret: " << ret;
    return ret;
  }

  auto cond_fg_cnode = cond_call_cnode->input(kCNodePrimIndex)->cast<CNodePtr>();
  MS_ASSERT(cond_fg_cnode != nullptr);
  AnfNodePtr cond_fg_vnode = cond_fg_cnode->input(kCNodeFirstInputIndex);
  MS_ASSERT(cond_fg_vnode != nullptr);
  auto cond_fg = GetValueNode<std::shared_ptr<FuncGraph>>(cond_fg_vnode);
  MS_CHECK_TRUE_MSG(cond_fg != nullptr, RET_FAILED, "Get value as func_graph failed.");

  CNodePtr body_partial_node = nullptr;
  ret = CreateWhileBodyPartialNode(cond_fg, while_cnode, &body_partial_node);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "while create body partial cnode failed, ret: " << ret;
    return ret;
  }

  CNodePtr after_partial_cnode = nullptr;
  ret = CreateWhileAfterPartialNode(fg, cond_fg, remain_nodes, visited_nodes_used_by_after_fg,
                                    visited_nodes_and_cond_fg_inputs_replace_pairs, &while_cnode, &after_partial_cnode);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "while create after partial cnode failed, ret: " << ret;
    return ret;
  }

  // create switch cnode
  ValueNodePtr switch_anf_primitive = lite::GetSwitchAnfPrim();
  if (switch_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetSwitchAnfPrim failed.";
    return lite::RET_ERROR;
  }

  // insert switch node
  std::vector<AnfNodePtr> switch_node_inputs = {switch_anf_primitive, cond_fg->output(), body_partial_node,
                                                after_partial_cnode};
  auto switch_cnode = cond_fg->NewCNode(switch_node_inputs);
  MS_CHECK_TRUE_MSG(switch_cnode != nullptr, RET_ERROR, "NewCnode failed");
  switch_cnode->set_fullname_with_scope("while-Switch-" + cond_fg->get_attr("graph_name")->ToString());

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{switch_cnode};
  auto call_node = cond_fg->NewCNode(call_node_inputs);
  MS_CHECK_TRUE_MSG(call_node != nullptr, lite::RET_NULL_PTR, "call_node is nullptr");
  call_node->set_fullname_with_scope("call_" + switch_cnode->fullname_with_scope());
  cond_fg->set_output(call_node);

  fg->DropNode(while_cnode);
  fg->set_output(cond_call_cnode);

  auto after_cnode = after_partial_cnode->input(kCNodeFirstInputIndex)->cast<ValueNodePtr>();
  MS_ASSERT(after_cnode != nullptr);
  auto after_fg = after_cnode->value()->cast<FuncGraphPtr>();
  if (after_fg == nullptr) {
    MS_LOG(ERROR) << "after_fg is nullptr.";
    return RET_FAILED;
  }
  to_process_q.push_back(cond_fg);
  to_process_q.push_back(after_fg);
  return RET_SUCCESS;
}

int ControlFlowPass::CreateIfPartialNodeExternalInputs(const CNodePtr &if_cnode, const FuncGraphPtr &partial_fg,
                                                       std::vector<AnfNodePtr> *then_partial_cnode_inputs) {
  auto if_inputs = if_cnode->inputs();
  auto fg_name_attr = partial_fg->get_attr("graph_name");
  MS_CHECK_TRUE_RET(fg_name_attr != nullptr, RET_FAILED);
  auto partial_fg_name = fg_name_attr->ToString();
  std::vector<AnfNodePtr> if_external_inputs{};
  if_external_inputs.assign(if_inputs.begin() + kIfMinInputSize, if_inputs.end());
  auto origin_then_fg_inputs = partial_fg->get_inputs();
  if (if_external_inputs.size() < origin_then_fg_inputs.size()) {
    MS_LOG(ERROR) << "graph is not right.";
    return RET_FAILED;
  } else if (if_external_inputs.size() == origin_then_fg_inputs.size()) {
    then_partial_cnode_inputs->insert(then_partial_cnode_inputs->end(), if_external_inputs.begin(),
                                      if_external_inputs.end());
    return RET_SUCCESS;
  } else {
    for (auto &fg_input : origin_then_fg_inputs) {
      auto fg_input_name = fg_input->fullname_with_scope();
      auto pos = partial_fg_name.size() + sizeof("_input_");
      auto pos2 = fg_input_name.find('_', pos);
      auto idx_str = fg_input_name.substr(pos - 1, pos2 - pos + 1);
      auto partial_idx = 0;
      try {
        partial_idx = std::stoi(idx_str);
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Get index failed: " << e.what();
        return RET_FAILED;
      }
      then_partial_cnode_inputs->push_back(if_external_inputs.at(partial_idx));
    }
  }
  return RET_SUCCESS;
}

int ControlFlowPass::CreateIfPartialNode(const FuncGraphPtr &fg, const size_t &index,
                                         std::vector<AnfNodePtr> *visited_nodes_used_by_after_fg,
                                         const CNodePtr &if_cnode, const FuncGraphPtr &after_fg,
                                         CNodePtr *then_partial_cnode) {
  auto then_vnode = if_cnode->input(index);
  MS_ASSERT(then_vnode != nullptr);
  auto then_fg = GetValueNode<std::shared_ptr<FuncGraph>>(then_vnode);
  MS_CHECK_TRUE_MSG(then_fg != nullptr, RET_FAILED, "Get value as func_graph failed.");

  // create then partial node
  ValueNodePtr then_partial_anf_primitive = lite::GetPartialFusionPrim();
  MS_CHECK_TRUE_MSG(then_partial_anf_primitive != nullptr, RET_FAILED, "GetPartialFusionPrim failed.");
  std::vector<AnfNodePtr> then_partial_cnode_inputs{then_partial_anf_primitive, then_vnode};
  if (CreateIfPartialNodeExternalInputs(if_cnode, then_fg, &then_partial_cnode_inputs) != RET_SUCCESS) {
    MS_LOG(ERROR) << "CreateIfPartialNodeExternalInputs failed.";
    return RET_FAILED;
  }
  std::unordered_map<AnfNodePtr, AnfNodePtr> visited_nodes_and_after_partial_inputs_replace_pairs{};
  std::vector<AnfNodePtr> then_nodes_used_by_after_partial{};
  // set fg inputs to then_partial_cnode inputs
  auto origin_then_fg_inputs = then_fg->get_inputs();
  for (auto &item : *visited_nodes_used_by_after_fg) {
    bool found = false;
    size_t input_index = 0;
    for (size_t i = kPartialFirstInputSize; i < then_partial_cnode_inputs.size(); ++i) {
      if (then_partial_cnode_inputs[i] == item) {
        found = true;
        input_index = i - kPartialFirstInputSize;
        break;
      }
    }
    if (found) {
      visited_nodes_and_after_partial_inputs_replace_pairs[item] = origin_then_fg_inputs.at(input_index);
      then_nodes_used_by_after_partial.push_back(origin_then_fg_inputs.at(input_index));
      continue;
    }

    // set after fg inputs to cond_partial_cnode inputs
    then_partial_cnode_inputs.push_back(item);
    auto new_parameter = then_fg->add_parameter();
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, RET_FAILED, "new_parameter is nullptr");
    if (index == kIfThenIndex) {
      new_parameter->set_name(item->fullname_with_scope() + "_then_fg_parameter");
    } else {
      new_parameter->set_name(item->fullname_with_scope() + "_else_fg_parameter");
    }
    new_parameter->set_abstract(item->abstract());
    visited_nodes_and_after_partial_inputs_replace_pairs[item] = new_parameter;
    then_nodes_used_by_after_partial.push_back(new_parameter);
  }
  *then_partial_cnode = fg->NewCNode(then_partial_cnode_inputs);
  MS_CHECK_TRUE_MSG(*then_partial_cnode != nullptr, RET_FAILED, "new cnode is nullptr");
  auto fg_name_attr = then_fg->get_attr("graph_name");
  MS_CHECK_TRUE_RET(fg_name_attr != nullptr, RET_FAILED);
  auto then_fg_name = fg_name_attr->ToString();
  (*then_partial_cnode)->set_fullname_with_scope("partial_" + then_fg_name);

  // create after partial node
  ValueNodePtr after_partial_anf_primitive = lite::GetPartialFusionPrim();
  MS_CHECK_TRUE_MSG(after_partial_anf_primitive != nullptr, RET_FAILED, "GetPartialFusionPrim failed.");
  auto after_value_node = NewValueNode(after_fg);
  MS_CHECK_TRUE_MSG(after_value_node != nullptr, RET_FAILED, "NewValueNode failed.");
  // make the right after partial input
  std::vector<AnfNodePtr> after_partial_cnode_inputs{after_partial_anf_primitive, after_value_node};
  if (!CheckPrimitiveType(then_fg->output(), prim::kPrimMakeTuple)) {
    after_partial_cnode_inputs.push_back(then_fg->output());
  } else {
    auto then_fg_output = then_fg->output()->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(then_fg_output != nullptr, RET_ERROR, "cast ptr failed");
    for (size_t i = kCNodeFirstInputIndex; i < then_fg_output->inputs().size(); ++i) {
      after_partial_cnode_inputs.push_back(then_fg_output->input(i));
    }
    then_fg->DropNode(then_fg_output);
  }
  size_t if_output_size = after_partial_cnode_inputs.size() - kCNodeSecondInputIndex;

  // add after fg inputs to partial node
  std::copy(then_nodes_used_by_after_partial.begin(), then_nodes_used_by_after_partial.end(),
            std::back_inserter(after_partial_cnode_inputs));
  // insert partial node
  auto after_partial_cnode = then_fg->NewCNode(after_partial_cnode_inputs);
  MS_CHECK_TRUE_MSG(after_partial_cnode != nullptr, RET_FAILED, "NewCNode failed");
  auto after_fg_name = after_fg->get_attr("graph_name")->ToString();
  after_partial_cnode->set_fullname_with_scope("partial_" + after_fg_name);

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{after_partial_cnode};
  auto call_node = then_fg->NewCNode(call_node_inputs);
  MS_CHECK_TRUE_MSG(call_node != nullptr, RET_FAILED, "NewCNode failed");
  call_node->set_fullname_with_scope("call_" + after_partial_cnode->fullname_with_scope());
  then_fg->set_output(call_node);
  to_process_q.push_back(then_fg);
  ReplaceNode(after_fg, visited_nodes_and_after_partial_inputs_replace_pairs);

  // check the inputs of after fg
  auto after_fg_inputs_size = after_fg->get_inputs().size();
  if (after_fg_inputs_size == after_partial_cnode_inputs.size() - kPartialFirstInputSize) {
    return RET_SUCCESS;
  }

  // make the inputs of the after fg
  std::unordered_map<AnfNodePtr, AnfNodePtr> after_partial_after_fg_replace_pairs{};
  for (size_t i = kPartialFirstInputSize; i < after_partial_cnode_inputs.size(); ++i) {
    auto &input = after_partial_cnode_inputs[i];
    auto new_parameter = after_fg->add_parameter();
    MS_CHECK_TRUE_MSG(new_parameter != nullptr, RET_FAILED, "add_parameter failed");
    new_parameter->set_name(std::to_string(i - kPartialFirstInputSize) + "_" + input->fullname_with_scope());
    new_parameter->set_abstract(input->abstract());
    if (i < kPartialFirstInputSize + if_output_size) {
      after_partial_after_fg_replace_pairs[if_cnode] = new_parameter;
    } else {
      after_partial_after_fg_replace_pairs[input] = new_parameter;
    }
  }
  ReplaceNode(after_fg, after_partial_after_fg_replace_pairs);

  return RET_SUCCESS;
}

int ControlFlowPass::CreateIfElsePartialNode(const FuncGraphPtr &main_fg,
                                             std::vector<AnfNodePtr> *visited_nodes_used_by_after_fg,
                                             const CNodePtr &if_cnode, const FuncGraphPtr &after_fg,
                                             CNodePtr *else_partial_cnode) {
  return CreateIfPartialNode(main_fg, kIfElseIndex, visited_nodes_used_by_after_fg, if_cnode, after_fg,
                             else_partial_cnode);
}

int ControlFlowPass::CreateIfThenPartialNode(const FuncGraphPtr &main_fg,
                                             std::vector<AnfNodePtr> *visited_nodes_used_by_after_fg,
                                             const CNodePtr &if_cnode, const FuncGraphPtr &after_fg,
                                             CNodePtr *then_partial_cnode) {
  return CreateIfPartialNode(main_fg, kIfThenIndex, visited_nodes_used_by_after_fg, if_cnode, after_fg,
                             then_partial_cnode);
}

int ControlFlowPass::ProcessIfOp(const FuncGraphPtr &fg, const std::set<AnfNodePtr> &visited_nodes,
                                 const std::vector<AnfNodePtr> &remain_nodes, const AnfNodePtr &if_node) {
  if (if_node == nullptr) {
    MS_LOG(INFO) << "not found if, no need to process.";
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
  std::vector<AnfNodePtr> visited_nodes_used_by_after_fg{};
  VisitedNodesUsedByAfterParts(visited_nodes, remain_nodes, &visited_nodes_used_by_after_fg);

  CNodePtr then_partial_cnode = nullptr;
  int ret = CreateIfThenPartialNode(fg, &visited_nodes_used_by_after_fg, if_cnode, after_fg, &then_partial_cnode);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "if create then partial cnode failed, ret: " << ret;
    return ret;
  }

  CNodePtr else_partial_cnode = nullptr;
  ret = CreateIfElsePartialNode(fg, &visited_nodes_used_by_after_fg, if_cnode, after_fg, &else_partial_cnode);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "if create else partial cnode failed, ret: " << ret;
    return ret;
  }

  // create switch cnode
  ValueNodePtr switch_anf_primitive = lite::GetSwitchAnfPrim();
  if (switch_anf_primitive == nullptr) {
    MS_LOG(ERROR) << "GetSwitchAnfPrim failed.";
    return RET_FAILED;
  }

  //  insert switch node
  std::vector<AnfNodePtr> switch_node_inputs = {switch_anf_primitive, if_cnode->input(kIfCondIndex), then_partial_cnode,
                                                else_partial_cnode};
  auto switch_cnode = fg->NewCNode(switch_node_inputs);
  MS_CHECK_TRUE_MSG(switch_cnode != nullptr, RET_FAILED, "NewCNode failed");
  switch_cnode->set_fullname_with_scope("if-Switch-" + fg->get_attr("graph_name")->ToString());

  // insert call node
  std::vector<AnfNodePtr> call_node_inputs{switch_cnode};
  auto call_node = fg->NewCNode(call_node_inputs);
  MS_CHECK_TRUE_MSG(call_node != nullptr, RET_FAILED, "NewCNode failed");
  call_node->set_fullname_with_scope("call_" + switch_cnode->fullname_with_scope());
  fg->DropNode(if_cnode);
  fg->set_output(call_node, true);

  to_process_q.push_back(after_fg);
  return RET_SUCCESS;
}

int ControlFlowPass::ProcessControlOp(const FuncGraphPtr &fg) {
  if (fg == nullptr) {
    MS_LOG(ERROR) << "fg is nullptr.";
    return RET_FAILED;
  }

  AnfNodePtr control_flow_node = nullptr;
  std::vector<AnfNodePtr> remain_nodes{};
  std::set<AnfNodePtr> visited_nodes{};
  int ret = SplitGraph(fg, &control_flow_node, &visited_nodes, &remain_nodes);
  if (ret != RET_SUCCESS) {
    MS_LOG(ERROR) << "SplitGraph failed, ret: " << ret;
    return ret;
  }

  if (control_flow_node == nullptr) {
    MS_LOG(INFO) << "not found control flow op, no need to process.";
    return RET_SUCCESS;
  }

  if (CheckPrimitiveType(control_flow_node, prim::kPrimWhile)) {
    ret = ProcessWhileOp(fg, visited_nodes, remain_nodes, control_flow_node);
    if (ret != RET_SUCCESS) {
      MS_LOG(ERROR) << "ProcessWhileOp failed.";
      return ret;
    }
  }

  if (CheckPrimitiveType(control_flow_node, prim::kPrimIf)) {
    ret = ProcessIfOp(fg, visited_nodes, remain_nodes, control_flow_node);
    if (ret != RET_SUCCESS) {
      MS_LOG(ERROR) << "ProcessIfOp failed.";
      return ret;
    }
  }
  return RET_SUCCESS;
}

bool ControlFlowPass::Run(const FuncGraphPtr &fg) {
  MS_ASSERT(fg != nullptr);
  to_process_q.push_back(fg);
  while (!to_process_q.empty()) {
    auto cur_fg = to_process_q.front();
    auto cur_fg_name = cur_fg->get_attr("graph_name")->ToString();
    int ret = ProcessControlOp(cur_fg);
    if (ret != RET_SUCCESS) {
      MS_LOG(ERROR) << "ProcessControlOp for graph: " << cur_fg_name << " failed.";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
      return false;
    }
    to_process_q.pop_front();
  }
  return true;
}
}  // namespace mindspore::opt
