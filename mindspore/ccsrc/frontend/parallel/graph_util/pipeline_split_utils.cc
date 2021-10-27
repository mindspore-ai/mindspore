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

#include <iterator>
#include <memory>
#include <list>
#include <set>
#include <algorithm>
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "base/core_ops.h"
#include "ir/value.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
const std::set<PrimitivePtr> END_NODE_BLACK_LIST = {
  prim::kPrimDepend,    prim::kPrimTupleGetItem, prim::kPrimAdd,    prim::kPrimSoftmaxCrossEntropyWithLogits,
  prim::kPrimMakeTuple, prim::kPrimUpdateState,  prim::kPrimReshape};

static bool IsInEndNodeBlackList(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return true;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (IsInParallelBlackList(prim)) {
    return true;
  }
  for (auto &prim_node : END_NODE_BLACK_LIST) {
    if (IsPrimitiveCNode(cnode, prim_node)) {
      return true;
    }
  }
  return false;
}

AnfNodePtr FindAccuGrad(const CNodePtr &cnode) {
  auto pre_node = cnode->input(1);
  size_t depth = 0;
  while (true) {
    if (depth > MAX_RECURSIVE_DEPTH) {
      return nullptr;
    }
    depth += 1;
    if (pre_node->isa<Parameter>()) {
      return pre_node;
    } else {
      if (pre_node->isa<CNode>()) {
        auto pre_cnode = pre_node->cast<CNodePtr>();
        pre_node = pre_cnode->input(1);
      } else {
        return nullptr;
      }
    }
  }
  return nullptr;
}

bool IsLastStage() {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto stage_num = g_device_manager->stage_num();
  auto stage_id = g_device_manager->stage_id();
  return ((stage_num - 1) == stage_id);
}

void SetStridedSliceStrategy(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<Shapes> shape_list = ExtractShape(cnode);
  if (shape_list.empty()) {
    MS_LOG(EXCEPTION) << "Failure:node " << cnode->ToString() << " failed to extract shape";
  }
  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < shape_list[0].size(); i++) {
    if (shape_list[0][i].empty()) {
      MS_LOG(EXCEPTION) << "shape_list[ " << i << " ].size() is zero";
    }
    Dimensions input_strategy;
    for (size_t j = 0; j < shape_list[0][i].size(); j++) {
      input_strategy.push_back(1);
    }
    elements.push_back(MakeValue(input_strategy));
  }
  ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
  cnode->AddPrimalAttr(STRATEGY, strategy);
}

void InsertVirtualAssignAdd(const std::pair<AnfNodePtr, int> &node_user, const FuncGraphManagerPtr &manager,
                            const AnfNodePtr &accu_parameter) {
  auto cnode = node_user.first->cast<CNodePtr>();
  if (IsPrimitiveCNode(cnode, prim::kPrimReceive) || !cnode->in_forward_flag()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool enable_parallel_optimizer = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && enable_parallel_optimizer) {
    return;
  }
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(WARNING) << cnode->DebugString() << " can not insert _VirtualAssignAdd.";
    return;
  }
  OperatorAttrs attrs;
  auto py_instance = CreatOpInstance(attrs, VIRTUAL_ASSIGN_ADD, VIRTUAL_ASSIGN_ADD);
  auto value_node = NewValueNode(py_instance);
  std::vector<AnfNodePtr> virtual_node_input = {value_node, cnode->input(IntToSize(node_user.second)), accu_parameter};
  auto graph = cnode->func_graph();
  auto virtual_node = graph->NewCNode(virtual_node_input);
  manager->SetEdge(cnode, node_user.second, virtual_node);
}

void InsertVirtualAccuGrad(const AnfNodePtr &recv, const FuncGraphManagerPtr &manager, const AnfNodePtr &param) {
  auto cnode = recv->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorAttrs attrs;
  auto py_instance = CreatOpInstance(attrs, VIRTUAL_ACCU_GRAD, VIRTUAL_ACCU_GRAD);
  auto value_node = NewValueNode(py_instance);
  std::vector<AnfNodePtr> virtual_node_input = {value_node, recv, param};
  auto graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto virtual_node = graph->NewCNode(virtual_node_input);
  (void)manager->Replace(recv, virtual_node);
}

AnfNodePtr FindGradAccuParameter(const std::vector<AnfNodePtr> &parameters, const std::string &name) {
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->name() == name) {
      continue;
    }
    auto expect_name = "accu_grads." + name;
    if (param_ptr->name() == expect_name) {
      return parameter;
    }
  }
  return nullptr;
}

void HandleReceiveParam(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  auto parameters = root->parameters();
  auto node_users_map = root->manager()->node_users();
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      continue;
    }
    auto parameter_ptr = cnode->input(1)->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter_ptr);
    auto accu_parameter = FindGradAccuParameter(parameters, parameter_ptr->name());
    if (!accu_parameter) {
      continue;
    }
    auto node_users = node_users_map[node];
    for (auto &temp_user : node_users) {
      auto temp_node = temp_user.first;
      // Micro virtual operator might be inserted after cast
      if (IsPrimitiveCNode(temp_node, prim::kPrimCast)) {
        temp_node = node_users_map[temp_node].begin()->first;
      }
      if (IsPrimitiveCNode(temp_node, prim::kPrimMirrorMicroStep) ||
          IsPrimitiveCNode(temp_node, prim::kPrimMicroStepAllGather)) {
        auto node_set = node_users_map[temp_node];
        for (auto &node_user : node_set) {
          InsertVirtualAssignAdd(node_user, root->manager(), accu_parameter);
        }
      } else {
        InsertVirtualAssignAdd(temp_user, root->manager(), accu_parameter);
      }
    }
    InsertVirtualAccuGrad(node, root->manager(), accu_parameter);
  }
}

void AddVirtualAssignAdd(const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  auto node_users_map = root->manager()->node_users();
  for (auto &parameter : parameters) {
    auto parameter_ptr = parameter->cast<ParameterPtr>();
    auto accu_parameter = FindGradAccuParameter(parameters, parameter_ptr->name());
    if (!accu_parameter) {
      continue;
    }
    auto node_users = node_users_map[parameter];
    for (auto &temp_user : node_users) {
      auto temp_node = temp_user.first;
      // Micro virtual operator might be inserted after cast
      if (IsPrimitiveCNode(temp_node, prim::kPrimCast)) {
        temp_node = node_users_map[temp_node].begin()->first;
      }
      if (IsPrimitiveCNode(temp_node, prim::kPrimMirrorMicroStep) ||
          IsPrimitiveCNode(temp_node, prim::kPrimMicroStepAllGather)) {
        auto node_set = node_users_map[temp_node];
        for (auto &node_user : node_set) {
          InsertVirtualAssignAdd(node_user, root->manager(), accu_parameter);
        }
      } else {
        InsertVirtualAssignAdd(temp_user, root->manager(), accu_parameter);
      }
    }
  }
}

bool CompFunc(const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  auto cnode1 = node1->cast<CNodePtr>();
  auto cnode2 = node2->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode1);
  MS_EXCEPTION_IF_NULL(cnode2);
  auto micro1 = cnode1->GetPrimalAttr(MICRO);
  auto micro2 = cnode2->GetPrimalAttr(MICRO);
  MS_EXCEPTION_IF_NULL(micro1);
  MS_EXCEPTION_IF_NULL(micro2);
  auto micro1_value = GetValue<int64_t>(micro1);
  auto micro2_value = GetValue<int64_t>(micro2);
  if (micro1_value == micro2_value) {
    auto prim1 = GetCNodePrimitive(cnode1);
    auto prim2 = GetCNodePrimitive(cnode2);
    MS_EXCEPTION_IF_NULL(prim1);
    MS_EXCEPTION_IF_NULL(prim2);
    auto rank_tag1 = prim1->GetAttr(SRC_RANK);
    auto rank_tag2 = prim2->GetAttr(SRC_RANK);
    if (rank_tag1 == nullptr) {
      rank_tag1 = prim1->GetAttr(DEST_RANK);
    }
    if (rank_tag2 == nullptr) {
      rank_tag2 = prim2->GetAttr(DEST_RANK);
    }
    if (!rank_tag1 || !rank_tag2) {
      return false;
    }
    auto rank1_value = GetValue<int64_t>(rank_tag1);
    auto rank2_value = GetValue<int64_t>(rank_tag2);
    if (rank1_value == rank2_value) {
      auto sr_tag1 = prim1->GetAttr(SR_TAG);
      auto sr_tag2 = prim2->GetAttr(SR_TAG);
      MS_EXCEPTION_IF_NULL(sr_tag1);
      MS_EXCEPTION_IF_NULL(sr_tag2);
      auto sr1_value = GetValue<int64_t>(sr_tag1);
      auto sr2_value = GetValue<int64_t>(sr_tag2);
      return sr1_value < sr2_value;
    }
    return rank1_value < rank2_value;
  }
  return micro1_value < micro2_value;
}

void InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node, const FuncGraphManagerPtr &manager,
                  const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node);
  auto post_cnode = post_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(post_cnode);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), post_cnode->input(1), prior_node};
  auto depend_node = root->NewCNode(depend_input);
  manager->SetEdge(post_node, 1, depend_node);
}

void ReorderForForward(const std::vector<AnfNodePtr> &forward_start, const std::vector<AnfNodePtr> &forward_end,
                       const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto stage_num = g_device_manager->stage_num();
  auto stage_id = g_device_manager->stage_id();
  for (size_t i = 1; i < LongToSize(stage_num - stage_id); ++i) {
    auto prior_node = forward_end[i - 1];
    auto post_node = forward_start[i];
    InsertDepend(prior_node, post_node, manager, root);
  }
}

void ReorderForBackward(const PipelinePair &forward_start_pair, const PipelinePair &forward_end_pair,
                        const PipelinePair &backward_start_pair, const PipelinePair &backward_end_pair,
                        const PipelinePair &forward_end_before_pair, const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  MS_EXCEPTION_IF_NULL(root);
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto stage_num = g_device_manager->stage_num();
  auto stage_id = g_device_manager->stage_id();
  for (size_t i = LongToSize(stage_num - stage_id); i < (forward_start_pair.first.size()); ++i) {
    auto prior_node1 = forward_end_before_pair.second[i];
    auto post_node1 = backward_start_pair.first[LongToSize(SizeToLong(i) - stage_num + stage_id + 1)];
    InsertDepend(prior_node1, post_node1, manager, root);
    auto prior_node2 = backward_end_pair.second[LongToSize(SizeToLong(i) - stage_num + stage_id)];
    auto post_node2 = forward_start_pair.first[i];
    InsertDepend(prior_node2, post_node2, manager, root);
  }
  for (size_t i = LongToSize(stage_num - stage_id); i < (forward_start_pair.first.size() + 1); ++i) {
    if (!IsLastStage()) {
      auto prior_node3 = backward_start_pair.second[LongToSize(SizeToLong(i) - stage_num + stage_id)];
      auto post_node3 = forward_end_pair.first[i - 1];
      InsertDepend(prior_node3, post_node3, manager, root);
      auto prior_node4 = forward_end_pair.second[i - 1];
      auto post_node4 = backward_end_pair.first[LongToSize(SizeToLong(i) - stage_num + stage_id)];
      InsertDepend(prior_node4, post_node4, manager, root);
    }
  }
  for (size_t j = LongToSize(SizeToLong(backward_start_pair.first.size()) - stage_num + stage_id + 1);
       j < backward_start_pair.first.size(); ++j) {
    auto prior_node5 = backward_end_pair.second[j - 1];
    auto post_node5 = backward_start_pair.first[j];
    InsertDepend(prior_node5, post_node5, manager, root);
  }
  if (!IsLastStage()) {
    auto prior_node6 = forward_end_before_pair.second[LongToSize(stage_num - 1 - stage_id)];
    auto post_node6 = backward_start_pair.first[0];
    InsertDepend(prior_node6, post_node6, manager, root);
  }
}

void ReorderForParams(const std::vector<AnfNodePtr> &backward_params, const std::vector<AnfNodePtr> &forward_params,
                      const std::vector<AnfNodePtr> &allreduce_params, const PipelinePair &forward_params_pair,
                      const PipelinePair &backward_params_pair, const std::vector<AnfNodePtr> &backward_end,
                      const PipelinePair &forward_start_pair, const FuncGraphPtr &root) {
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!forward_params.empty()) {
    auto prior_node = forward_params_pair.second[0];
    auto post_node = forward_start_pair.first[0];
    InsertDepend(prior_node, post_node, manager, root);
  }
  if (!backward_params.empty()) {
    if (!allreduce_params.empty()) {
      for (auto &node : allreduce_params) {
        auto post_node1 = backward_params_pair.first[0];
        InsertDepend(node, post_node1, manager, root);
      }
    }
    auto prior_node2 = backward_end.back();
    auto post_node2 = backward_params[0];
    InsertDepend(prior_node2, post_node2, manager, root);
  }
}

int64_t GetMicroBatch(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto micro_value = cnode->GetPrimalAttr(MICRO);
  MS_EXCEPTION_IF_NULL(micro_value);
  return GetValue<int64_t>(micro_value);
}

PipelinePair Deduplicate(const std::vector<AnfNodePtr> &node_vector, const FuncGraphPtr &root, int64_t micro_max) {
  std::vector<AnfNodePtr> temp_vec;
  std::vector<AnfNodePtr> out_vec_begin;
  std::vector<AnfNodePtr> out_vec_end;
  auto manager = root->manager();
  for (int64_t i = 0; i <= micro_max; ++i) {
    temp_vec.clear();
    if (!root->has_flag(TRAINING)) {
      temp_vec = node_vector;
    } else {
      for (auto &node : node_vector) {
        auto node_micro = GetMicroBatch(node);
        if (node_micro == i) {
          temp_vec.push_back(node);
        }
      }
    }
    if (temp_vec.size() <= 1) {
      MS_LOG(INFO) << "No Duplicate MicroBatch.";
      continue;
    }
    std::sort(temp_vec.begin(), temp_vec.end(), CompFunc);
    for (size_t j = 0; j < temp_vec.size() - 1; ++j) {
      auto prior_node = temp_vec[j];
      auto post_node = temp_vec[j + 1];
      InsertDepend(prior_node, post_node, manager, root);
    }
    if (!temp_vec.empty()) {
      out_vec_begin.push_back(temp_vec.front());
      out_vec_end.push_back(temp_vec.back());
    }
  }
  if (out_vec_begin.empty()) {
    return std::make_pair(node_vector, node_vector);
  }
  return std::make_pair(out_vec_begin, out_vec_end);
}

void BroadCastMicroBatch(const CNodePtr &node, NodeUsersMap *node_users_map, const ValuePtr &value, size_t max_depth) {
  auto node_users = (*node_users_map)[node];
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(EXCEPTION) << "Recursive call is larger than 100000.";
  }
  for (auto &node_pair : node_users) {
    auto user_node = node_pair.first->cast<CNodePtr>();
    if (user_node->HasPrimalAttr(MICRO)) {
      continue;
    }
    user_node->AddPrimalAttr(MICRO, value);
    BroadCastMicroBatch(user_node, node_users_map, value, max_depth + 1);
  }
}

void BroadCastNeedGrad(const AnfNodePtr &node, NodeUsersMap *node_user_map, const FuncGraphPtr &root) {
  auto node_users = (*node_user_map)[node];
  for (auto &node_user : node_users) {
    auto cnode = node_user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->HasPrimalAttr(NEED_GRAD)) {
      continue;
    }
    if (cnode->func_graph() == root) {
      continue;
    }
    cnode->AddPrimalAttr(NEED_GRAD, MakeValue(1));
    BroadCastNeedGrad(cnode, node_user_map, root);
  }
}

// Label node that need backpropagation
void LabelNeedGrad(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  auto node_user_map = manager->node_users();
  for (auto &parameter : parameters) {
    if (!ParameterRequireGrad(parameter)) {
      continue;
    }
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->name().find(ACCU_GRADS) != std::string::npos) {
      continue;
    }
    BroadCastNeedGrad(parameter, &node_user_map, root);
  }
}

AnfNodePtr GetPreNode(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> node_queue = {node};
  while (!node_queue.empty()) {
    auto cur_node = (*node_queue.begin())->cast<CNodePtr>();
    if (!cur_node) {
      (void)node_queue.erase(node_queue.begin());
      continue;
    }
    (void)node_queue.erase(node_queue.begin());
    if (!IsInEndNodeBlackList(cur_node) && cur_node->HasPrimalAttr(NEED_GRAD)) {
      MS_LOG(INFO) << "Pipeline End node: " << cur_node->DebugString();
      return cur_node;
    }
    (void)node_queue.insert(node_queue.end(), cur_node->inputs().begin() + 1, cur_node->inputs().end());
  }
  MS_LOG(EXCEPTION) << "Get Pipeline End node failed.";
}

void LastStageEndNode(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager,
                      const FuncGraphPtr &root) {
  if (!IsLastStage()) {
    return;
  }
  LabelNeedGrad(manager, root);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!cnode->HasPrimalAttr(MICRO)) {
      continue;
    }
    auto prim = GetCNodePrimitive(node);
    if (prim && prim->HasAttr(PIPELINE_END)) {
      for (auto &temp_node : cnode->inputs()) {
        if (!temp_node->isa<CNode>()) {
          continue;
        }
        auto temp_prim = GetCNodePrimitive(temp_node);
        if (!temp_prim || temp_prim->HasAttr(PIPELINE_END)) {
          continue;
        }
        auto end_node = GetPreNode(temp_node);
        MS_EXCEPTION_IF_NULL(end_node);
        auto end_cnode = end_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(end_cnode);
        auto end_prim = GetCNodePrimitive(end_node);
        OperatorAttrs attrs_;
        auto op = CreatOpInstance(attrs_, end_prim->name(), "");
        auto value_node = NewValueNode(op);
        auto new_prim = GetValueNode(value_node)->cast<PrimitivePtr>();
        (void)new_prim->SetAttrs(end_prim->attrs());
        manager->SetEdge(end_node, 0, value_node);
        end_cnode->AddPrimalAttr(PIPELINE_END, end_cnode->GetPrimalAttr(MICRO));
      }
    }
  }
}

ValuePtr Micro(const CNodePtr &cnode, NodeUsersMap *node_users_map, size_t max_depth) {
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(EXCEPTION) << "Recursive call is larger than 100000.";
  }
  if (cnode->HasPrimalAttr(MICRO)) {
    return cnode->GetPrimalAttr(MICRO);
  }
  auto node_users = (*node_users_map)[cnode];
  for (auto &node_pair : node_users) {
    auto user_node = node_pair.first->cast<CNodePtr>();
    auto micro = Micro(user_node, node_users_map, max_depth + 1);
    if (micro) {
      return micro;
    }
  }
  return nullptr;
}

void ParameterStartNode(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  auto node_users_map = manager->node_users();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(node);
    if (prim && prim->HasAttr(PARAMETER_START)) {
      auto micro = Micro(cnode, &node_users_map, 0);
      cnode->AddPrimalAttr(MICRO, micro);
      cnode->AddPrimalAttr(PARAMETER_START, micro);
    }
  }
}

void HandleMicroBatch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  auto node_users_map = manager->node_users();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!cnode->HasPrimalAttr(MICRO)) {
      continue;
    }
    auto micro = cnode->GetPrimalAttr(MICRO);
    MS_EXCEPTION_IF_NULL(micro);
    BroadCastMicroBatch(cnode, &node_users_map, micro, 0);
  }
}

AnfNodePtr GetActualOp(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    auto cnode = node->cast<CNodePtr>();
    return cnode->input(1);
  }
  return node;
}

void GetBorderNode(std::vector<AnfNodePtr> *forward_start, std::vector<AnfNodePtr> *forward_end,
                   std::vector<AnfNodePtr> *backward_start, std::vector<AnfNodePtr> *backward_end,
                   std::vector<AnfNodePtr> *forward_params, std::vector<AnfNodePtr> *backward_params,
                   std::vector<AnfNodePtr> *allreduce_params, const FuncGraphPtr &root) {
  std::list<ValuePtr> name_list = {};
  auto stage_id = g_device_manager->stage_id();
  for (auto &node : root->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimZerosLike)) {
      continue;
    }
    auto prim = GetCNodePrimitive(node);
    auto cnode = node->cast<CNodePtr>();
    if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      auto forward_node_name = cnode->GetPrimalAttr(kPrimalAttrForwardNodeName);
      if (std::find(name_list.begin(), name_list.end(), forward_node_name) != name_list.end()) {
        continue;
      }
      name_list.push_back(forward_node_name);
      if (cnode->HasPrimalAttr(PIPELINE_END)) {
        backward_start->push_back(node);
      }
      if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
        backward_end->push_back(node);
      }
      if (cnode->HasPrimalAttr(PARAMETER_START)) {
        backward_end->push_back(node);
      }
      if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
        backward_params->push_back(node);
      }
      if (prim->HasAttr(PARAMETER_MICRO)) {
        allreduce_params->push_back(node);
      }
    } else {
      if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
        if (stage_id != 0 && IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
          continue;
        }
        forward_start->push_back(node);
      }
      if (cnode->HasPrimalAttr(PIPELINE_END)) {
        forward_end->push_back(node);
      }
      if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
        forward_params->push_back(node);
      }
    }
  }
  std::sort((*backward_start).begin(), (*backward_start).end(), CompFunc);
  std::sort((*backward_end).begin(), (*backward_end).end(), CompFunc);
  std::sort((*forward_start).begin(), (*forward_start).end(), CompFunc);
  std::sort((*forward_end).begin(), (*forward_end).end(), CompFunc);
  std::sort((*backward_params).begin(), (*backward_params).end(), CompFunc);
  std::sort((*forward_params).begin(), (*forward_params).end(), CompFunc);
}

void CheckBorderNode(const PipelinePair &forward_start_pair, const PipelinePair &forward_end_pair,
                     const PipelinePair &backward_start_pair, const PipelinePair &backward_end_pair,
                     size_t micro_size) {
  micro_size = micro_size + 1;
  if (forward_start_pair.first.size() != micro_size) {
    MS_LOG(EXCEPTION) << "forward_node's size:" << forward_start_pair.first.size()
                      << "is not equal to micro size:" << micro_size;
  }
  if (forward_end_pair.first.size() != micro_size) {
    MS_LOG(EXCEPTION) << "forward_node's size:" << forward_end_pair.first.size()
                      << "is not equal to micro size:" << micro_size;
  }
  if (backward_start_pair.first.size() != micro_size) {
    MS_LOG(EXCEPTION) << "backward_node's size:" << backward_start_pair.first.size()
                      << "is not equal to micro size:" << micro_size;
  }
  if (backward_end_pair.first.size() != micro_size) {
    MS_LOG(EXCEPTION) << "backward_node's size:" << backward_end_pair.first.size()
                      << "is not equal to micro size:" << micro_size;
  }
}

void Reorder(const FuncGraphPtr &root) {
  std::vector<AnfNodePtr> forward_start;
  std::vector<AnfNodePtr> forward_end;
  std::vector<AnfNodePtr> forward_params;
  std::vector<AnfNodePtr> backward_start;
  std::vector<AnfNodePtr> backward_end;
  std::vector<AnfNodePtr> backward_params;
  std::vector<AnfNodePtr> allreduce_params;
  GetBorderNode(&forward_start, &forward_end, &backward_start, &backward_end, &forward_params, &backward_params,
                &allreduce_params, root);
  int64_t micro_max = 0;
  if (root->has_flag(TRAINING)) {
    auto forward_end_cnode = forward_end.back()->cast<CNodePtr>();
    auto micro_size = forward_end_cnode->GetPrimalAttr(MICRO);
    MS_EXCEPTION_IF_NULL(micro_size);
    micro_max = GetValue<int64_t>(micro_size);
  }
  auto backward_start_pair = Deduplicate(backward_start, root, micro_max);
  auto backward_end_pair = Deduplicate(backward_end, root, micro_max);
  auto forward_start_pair = Deduplicate(forward_start, root, micro_max);
  auto forward_end_pair = Deduplicate(forward_end, root, micro_max);
  auto forward_params_pair = Deduplicate(forward_params, root, micro_max);
  auto backward_params_pair = Deduplicate(backward_params, root, micro_max);
  CheckBorderNode(forward_start_pair, forward_end_pair, backward_start_pair, backward_end_pair, LongToSize(micro_max));
  PipelinePair forward_end_before_pair;
  if (!IsLastStage()) {
    for (auto &node : forward_end_pair.first) {
      auto cnode = node->cast<CNodePtr>();
      auto temp_node = GetActualOp(cnode->input(1));
      MS_EXCEPTION_IF_NULL(temp_node);
      forward_end_before_pair.first.push_back(temp_node);
    }
    for (auto &node : forward_end_pair.second) {
      auto cnode = node->cast<CNodePtr>();
      auto temp_node = GetActualOp(cnode->input(1));
      MS_EXCEPTION_IF_NULL(temp_node);
      forward_end_before_pair.second.push_back(temp_node);
    }
  } else {
    forward_end_before_pair = forward_end_pair;
  }
  ReorderForForward(forward_start_pair.first, forward_end_pair.second, root);
  ReorderForBackward(forward_start_pair, forward_end_pair, backward_start_pair, backward_end_pair,
                     forward_end_before_pair, root);
  ReorderForParams(backward_params, forward_params, allreduce_params, forward_params_pair, backward_params_pair,
                   backward_end, forward_start_pair, root);
}

void ReorderForPredict(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  std::vector<AnfNodePtr> forward_end;
  std::vector<AnfNodePtr> forward_start;
  std::vector<AnfNodePtr> forward_params;
  for (auto &node : root->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
      forward_start.push_back(node);
    }
    if (cnode->HasPrimalAttr(PIPELINE_END)) {
      forward_end.push_back(node);
    }
    if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      forward_params.push_back(node);
    }
  }
  std::sort(forward_start.begin(), forward_start.end(), CompFunc);
  std::sort(forward_end.begin(), forward_end.end(), CompFunc);
  std::sort(forward_params.begin(), forward_params.end(), CompFunc);
  auto forward_start_pair = Deduplicate(forward_start, root, 0);
  auto forward_end_pair = Deduplicate(forward_end, root, 0);
  auto forward_params_pair = Deduplicate(forward_params, root, 0);
  if (!forward_end.empty() && !forward_params.empty()) {
    InsertDepend(forward_params_pair.second[0], forward_end_pair.first[0], manager, root);
  }
  if (!forward_start.empty() && !forward_params.empty()) {
    InsertDepend(forward_params_pair.second[0], forward_start_pair.first[0], manager, root);
  }
}
}  // namespace parallel
}  // namespace mindspore
