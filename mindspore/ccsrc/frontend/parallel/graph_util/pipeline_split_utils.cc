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

#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include <algorithm>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/fold_pipeline_split_utils.h"
#include "include/common/utils/parallel_context.h"
#include "ir/value.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "ops/other_ops.h"
#include "ops/sequence_ops.h"
#include "utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
namespace {
bool IsSendRec(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimSend) || IsPrimitiveCNode(node, prim::kPrimReceive);
}

std::string TagForSendRecDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node) {
  if (!IsSendRec(prior_node) || !IsSendRec(post_node)) {
    return "";
  }
  if (prior_node->cast<CNodePtr>()->HasPrimalAttr(kPrimalAttrForwardNodeName) ==
      post_node->cast<CNodePtr>()->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
    return "";
  }
  return std::string(SEND_REC_DEPEND);
}
}  // namespace

bool IsFirstStage() {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto stage_id = g_device_manager->stage_id();
  return stage_id == 0;
}

bool IsLastStage() {
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto stage_num = g_device_manager->stage_num();
  auto stage_id = g_device_manager->stage_id();
  return ((stage_num - 1) == stage_id);
}

static ValuePtr GetReceiveMicro(const CNodePtr &cnode) {
  std::queue<CNodePtr> que;
  std::set<AnfNodePtr> visited;
  que.push(cnode);
  while (!que.empty()) {
    auto front = que.front();
    que.pop();
    (void)(visited.insert(front));
    for (size_t i = 1; i < front->inputs().size(); ++i) {
      auto input = front->input(i);
      if (!input->isa<CNode>()) {
        continue;
      }
      auto cinput = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cinput);
      if (IsPrimitiveCNode(cinput, prim::kPrimReceive)) {
        return cinput->GetPrimalAttr(MICRO);
      }
      if (visited.find(cinput) == visited.end()) {
        que.push(cinput);
      }
    }
  }
  return nullptr;
}

static ValuePtr GetReceiveSegment(const CNodePtr &cnode) {
  std::queue<CNodePtr> que;
  std::set<AnfNodePtr> visited;
  que.push(cnode);
  while (!que.empty()) {
    auto front = que.front();
    que.pop();
    (void)(visited.insert(front));
    for (size_t i = 1; i < front->inputs().size(); ++i) {
      auto input = front->input(i);
      if (!input->isa<CNode>()) {
        continue;
      }
      auto cinput = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cinput);
      if (IsPrimitiveCNode(cinput, prim::kPrimReceive)) {
        return cinput->GetPrimalAttr(SEGMENT);
      }
      if (visited.find(cinput) == visited.end()) {
        que.push(cinput);
      }
    }
  }
  return nullptr;
}

static bool EnableShareCell() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  const auto &comm_reuse_env = common::GetEnv("MS_COMM_COMPILER_OPT");
  if (!comm_reuse_env.empty() && cell_reuse) {
    MS_LOG(EXCEPTION) << "The cell reuse cannot be used with communication reuse,"
                         " please unset environment variable 'MS_COMM_COMPILER_OPT'";
  }
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool grad_accumulation_shard = ParallelContext::GetInstance()->grad_accumulation_shard();
  if (grad_accumulation_shard && cell_reuse) {
    MS_LOG(EXCEPTION)
      << "The cell reuse cannot be used with sharding accumulate grad parameter with optimizer parallel,"
         " please set_auto_parallel_context(parallel_optimizer_config={'gradient_accumulation_shard':False})";
  }
  return cell_reuse;
}

static AnfNodePtr GetCallBackwardEndNext(const AnfNodePtr &node) {
  if (!node->has_user_data(CALL_BACKWARD_END_NEXT)) {
    return node;
  }
  return node->user_data<AnfNode>(CALL_BACKWARD_END_NEXT);
}

bool IsValidNode(const AnfNodePtr &node, const AnfNodePtr &return_node, const NodeUsersMap &node_user_map) {
  if (node == return_node) {
    return true;
  }
  auto iter = node_user_map.find(node);
  if (iter == node_user_map.end()) {
    return false;
  }
  const auto &users = (*iter).second;
  return std::any_of(users.begin(), users.end(),
                     [&return_node, &node_user_map](const std::pair<AnfNodePtr, int> &user) {
                       return IsValidNode(user.first, return_node, node_user_map);
                     });
}

// judge if the graph call specified grad nodes, the specified grad nodes is in grad_graph
// search if call specified grad nodes according to dfs
static bool CallGradNodes(const FuncGraphPtr &graph, const FuncGraphPtr &grad_graph,
                          std::set<FuncGraphPtr> *const visit) {
  if (visit->find(graph) != visit->end()) {
    return false;
  }
  if (graph == grad_graph) {
    return true;
  }
  (void)(visit->insert(graph));
  const auto &cnodes = graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    const auto &abs = cnode->input(0)->abstract();
    if (!abs || !abs->isa<abstract::AbstractFunction>()) {
      continue;
    }
    const auto &abs_func = abs->cast<abstract::AbstractFunctionPtr>();
    if (!abs_func->isa<abstract::FuncGraphAbstractClosure>()) {
      continue;
    }
    const auto &abs_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    auto fg = abs_func_graph->func_graph();
    if (fg && fg == grad_graph) {
      return true;
    }
    if (CallGradNodes(fg, grad_graph, visit)) {
      return true;
    }
  }
  return false;
}

static FuncGraphPtr FindGradGraph(const FuncGraphPtr &root) {
  const auto &nodes = DeepScopedGraphSearch(root->get_return());
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    if (cnode->HasPrimalAttr(PARAMETER_START_SHARE_CELL) && cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      const auto &grad_graph = cnode->func_graph();
      MS_LOG(INFO) << "The specified grad nodes is in graph " << grad_graph->ToString();
      return grad_graph;
    }
  }
  MS_LOG(EXCEPTION) << "Stage0: The grad graph has not been found in lazy inline mode.";
  return nullptr;
}

void SetParameterStartForCellShare(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto share_cell = EnableShareCell();
  if (!share_cell) {
    return;
  }
  if (!IsFirstStage()) {
    return;
  }
  FuncGraphPtr grad_graph = FindGradGraph(root);
  MS_EXCEPTION_IF_NULL(grad_graph);
  const auto &manager = root->manager();
  auto node_user_map = manager->node_users();
  auto all_nodes = root->GetOrderedCnodes();
  std::set<FuncGraphPtr> call_grad_nodes;
  bool has_find = false;
  for (auto &node : all_nodes) {
    // if cnode is a call_backward node
    if (!IsPrimitiveCNode(node->input(0), prim::kPrimTupleGetItem)) {
      continue;
    }
    const auto &abs = node->input(0)->abstract();
    if (!abs || !abs->isa<abstract::AbstractFunction>()) {
      continue;
    }
    const auto &abs_func = abs->cast<abstract::AbstractFunctionPtr>();
    if (!abs_func->isa<abstract::FuncGraphAbstractClosure>()) {
      continue;
    }
    std::set<FuncGraphPtr> visit;
    const auto &abs_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    auto fg = abs_func_graph->func_graph();
    if (!fg || (call_grad_nodes.find(fg) == call_grad_nodes.end() && !CallGradNodes(fg, grad_graph, &visit))) {
      continue;
    }
    if (call_grad_nodes.find(fg) == call_grad_nodes.end()) {
      (void)(call_grad_nodes.insert(fg));
    }
    auto micro = GetReceiveMicro(node);
    MS_EXCEPTION_IF_NULL(micro);
    auto node_abs = node->abstract();
    if (node_abs->isa<abstract::AbstractTuple>()) {
      CNodePtr next = nullptr;
      const auto &users = node_user_map[node];
      for (const auto &user : users) {
        const auto &cuser = user.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cuser);
        if (IsPrimitiveCNode(cuser, prim::kPrimTupleGetItem) && IsValidNode(cuser, root->get_return(), node_user_map)) {
          next = cuser;
          break;
        }
      }
      node->set_user_data<AnfNode>(CALL_BACKWARD_END_NEXT, next);
    }
    has_find = true;
    node->AddPrimalAttr(MICRO, micro);
    node->AddPrimalAttr(PARAMETER_START, micro);
    auto parallel_context = parallel::ParallelContext::GetInstance();
    if (parallel_context->enable_fold_pipeline()) {
      auto segment = GetReceiveSegment(node);
      MS_EXCEPTION_IF_NULL(segment);
      node->AddPrimalAttr(SEGMENT, segment);
    }
  }
  if (!has_find) {
    MS_LOG(EXCEPTION) << "Stage0: The backward end flag has not been marked in lazy inline mode.";
  } else {
    MS_LOG(INFO) << "Stage0: The backward end flag has been marked in lazy inline mode.";
  }
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

void SetStridedSliceStrategy(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
    return;
  }
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  auto dev_num = g_device_manager->stage_device_num();
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
    static const auto skip_redis = (common::GetEnv("PIPELINE_SLICE_SKIP_REDISTRIBUTION") == "1");
    if (skip_redis && !full_batch && input_strategy.size() > 0) {
      input_strategy[0] = dev_num < shape_list[1][0][0] ? dev_num : shape_list[1][0][0];
      auto prim = GetCNodePrimitive(node);
      if (prim->HasAttr("out_shard_size")) {
        auto out_shard_size = GetValue<int64_t>(prim->GetAttr("out_shard_size"));
        input_strategy[0] = out_shard_size;
      }
      auto attrs = prim->attrs();
      attrs[parallel::SKIP_REDISTRIBUTION] = MakeValue<bool>(true);
      (void)prim->SetAttrs(attrs);
    }

    elements.push_back(MakeValue(input_strategy));
  }
  ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
  cnode->AddPrimalAttr(IN_STRATEGY, strategy);
}

CNodePtr FindNodeWithMircoSize(const AnfNodePtr &node_user, const NodeUsersMap &node_users_map) {
  // Recursively find micro tags, this may takes much more time if layers are too much
  std::queue<AnfNodePtr> visited;
  visited.push(node_user);
  while (!visited.empty()) {
    auto cur_node = visited.front();
    visited.pop();
    if (node_users_map.find(cur_node) == node_users_map.end()) {
      continue;
    }
    auto users = node_users_map.at(cur_node);
    for (auto &temp_user : users) {
      auto cnode = temp_user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!cnode->HasPrimalAttr(MICRO)) {
        visited.push(temp_user.first);
      } else {
        return cnode;
      }
    }
  }
  return nullptr;
}

bool IsSourceUsedByMirror(const CNodePtr &node, const NodeUsersMap &node_user_map) {
  if (node->inputs().size() < 2) {
    return false;
  }
  auto parameter_node = node->input(1);
  if (parameter_node->cast<ParameterPtr>()) {
    for (auto &item : node_user_map.at(parameter_node)) {
      if (IsPrimitiveCNode(item.first, prim::kPrimMirrorMicroStep)) {
        return true;
      }
    }
  }
  return false;
}
void InsertVirtualAssignAdd(const std::pair<AnfNodePtr, int> &node_user, const FuncGraphManagerPtr &manager,
                            const AnfNodePtr &accu_parameter, const NodeUsersMap &node_user_map) {
  auto cnode = node_user.first->cast<CNodePtr>();
  if (IsPrimitiveCNode(cnode, prim::kPrimReceive) || IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) ||
      !cnode->in_forward_flag()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool enable_parallel_optimizer = ParallelContext::GetInstance()->enable_parallel_optimizer();
  bool grad_accumulation_shard = ParallelContext::GetInstance()->grad_accumulation_shard();
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && enable_parallel_optimizer &&
      IsSourceUsedByMirror(cnode, node_user_map)) {
    return;
  }
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(WARNING) << cnode->DebugString() << " can not insert _VirtualAssignAdd.";
    return;
  }
  auto param_ptr = accu_parameter->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param_ptr);
  // If grad_accumulation_shard is ture, a ReduceScatter will be inserted at each micro step,
  // So the fusion id should be different for each micro step
  // otherwise they will be fused into the one ReduceScatter alone micro_steps.
  // if grad_accumulation_shard is false, we pass an empty group, so no ReduceScatter will be inserted
  ValuePtr args1 = nullptr;
  ValuePtr args2 = nullptr;
  ValuePtr micro = nullptr;
  int64_t step = 0;
  if (grad_accumulation_shard) {
    auto cnode_with_micro_size = FindNodeWithMircoSize(cnode, node_user_map);
    if (cnode_with_micro_size && cnode_with_micro_size->HasPrimalAttr(MICRO)) {
      micro = cnode_with_micro_size->GetPrimalAttr(MICRO);
      step = GetValue<int64_t>(micro);
    }
  }
  args1 = MakeValue(param_ptr->user_data<TensorLayout>()->opt_shard_group());
  args2 = MakeValue(LongToSize(param_ptr->param_info()->comm_fusion()) + LongToSize(step) * PIPELINE_FUSTION_OFFSET);
  OperatorAttrs attrs = {};
  auto py_instance = CreateOpInstance(attrs, VIRTUAL_ASSIGN_ADD, VIRTUAL_ASSIGN_ADD);
  auto value_node = NewValueNode(py_instance);
  // Set the attribute of the reduce scatter
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(new_prim);
  auto attrs_prim = new_prim->attrs();
  attrs_prim[GROUP] = args1;
  attrs_prim[kAttrFusion] = args2;
  (void)new_prim->SetAttrs(attrs_prim);

  std::vector<AnfNodePtr> virtual_node_input = {value_node, cnode->input(IntToSize(node_user.second)), accu_parameter};
  auto graph = cnode->func_graph();
  auto virtual_node = graph->NewCNode(virtual_node_input);
  manager->SetEdge(cnode, node_user.second, virtual_node);
}

void InsertVirtualAccuGrad(const AnfNodePtr &recv, const FuncGraphManagerPtr &manager, const AnfNodePtr &param) {
  auto cnode = recv->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorAttrs attrs;
  auto py_instance = CreateOpInstance(attrs, VIRTUAL_ACCU_GRAD, VIRTUAL_ACCU_GRAD);
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

// If the graph likes the followings:
// 1. MicroStepAllGather->MirrorMicro->load, we need to visit the param after the load
std::vector<std::pair<AnfNodePtr, int>> FindNextNode(const std::pair<AnfNodePtr, int> &node_ptr,
                                                     const NodeUsersMap &node_users_map) {
  std::vector<std::pair<AnfNodePtr, int>> to_be_visited_set;
  if (!IsSomePrimitiveList(
        node_ptr.first->cast<CNodePtr>(),
        {prim::kPrimMirrorMicroStep->name(), prim::kPrimMicroStepAllGather->name(), prim::kPrimLoad->name()})) {
    (void)to_be_visited_set.emplace_back(node_ptr);
    return to_be_visited_set;
  }
  auto node_set = node_users_map.at(node_ptr.first);
  std::queue<std::pair<std::shared_ptr<AnfNode>, int>> visited;
  for (auto &node_user : node_set) {
    visited.push(node_user);
  }
  while (visited.size() >= 1) {
    auto node = visited.front();
    visited.pop();
    if (!IsSomePrimitiveList(
          node.first->cast<CNodePtr>(),
          {prim::kPrimMirrorMicroStep->name(), prim::kPrimMicroStepAllGather->name(), prim::kPrimLoad->name()})) {
      (void)to_be_visited_set.emplace_back(node);
    } else {
      auto next_node_set = node_users_map.at(node.first);
      for (auto &node_user : next_node_set) {
        visited.push(node_user);
      }
    }
  }
  return to_be_visited_set;
}

std::set<std::pair<AnfNodePtr, int>> FuncNodeUsersSet(const AnfNodePtr &parameter) {
  MS_EXCEPTION_IF_NULL(parameter->func_graph());
  MS_EXCEPTION_IF_NULL(parameter->func_graph()->manager());
  auto node_users_map = parameter->func_graph()->manager()->node_users();
  auto node_users = node_users_map[parameter];
  std::set<std::pair<AnfNodePtr, int>> all_node_users;
  for (auto &n_pair : node_users) {
    auto users_skip_virtual_nodes = FindNextNode(n_pair, node_users_map);
    for (const auto &node_pair : users_skip_virtual_nodes) {
      auto func_node_users = FuncGraphNodeUsers(node_pair);
      if (func_node_users.empty()) {
        (void)all_node_users.insert(node_pair);
        continue;
      }
      for (const auto &func_node_user : func_node_users) {
        all_node_users.insert(func_node_user);
      }
    }
  }
  return all_node_users;
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
    std::set<std::pair<AnfNodePtr, int>> all_node_users = FuncNodeUsersSet(node);
    for (auto &temp_user : all_node_users) {
      auto temp_node = temp_user.first;
      // Micro virtual operator might be inserted after cast
      if (IsPrimitiveCNode(temp_node, prim::kPrimCast)) {
        temp_node = node_users_map[temp_node].begin()->first;
      }
      if (IsPrimitiveCNode(temp_node, prim::kPrimMirrorMicroStep) ||
          IsPrimitiveCNode(temp_node, prim::kPrimMicroStepAllGather)) {
        auto node_set = node_users_map[temp_node];
        for (auto &node_user : node_set) {
          InsertVirtualAssignAdd(node_user, root->manager(), accu_parameter, node_users_map);
        }
      } else {
        InsertVirtualAssignAdd(temp_user, root->manager(), accu_parameter, node_users_map);
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
    std::set<std::pair<AnfNodePtr, int>> all_node_users = FuncNodeUsersSet(parameter);
    for (auto &temp_user : all_node_users) {
      // Micro virtual operator might be inserted after cast
      auto temp_node = temp_user;
      if (IsPrimitiveCNode(temp_node.first, prim::kPrimCast)) {
        temp_node = *node_users_map[temp_node.first].begin();
      }
      if (!IsSomePrimitiveList(
            temp_node.first->cast<CNodePtr>(),
            {prim::kPrimMirrorMicroStep->name(), prim::kPrimMicroStepAllGather->name(), prim::kPrimLoad->name()})) {
        InsertVirtualAssignAdd(temp_node, root->manager(), accu_parameter, node_users_map);
        continue;
      }
      auto node_set = FindNextNode(temp_node, node_users_map);
      for (auto &node_user : node_set) {
        InsertVirtualAssignAdd(node_user, root->manager(), accu_parameter, node_users_map);
      }
    }
  }
}

bool SliceSort(const CNodePtr &cnode1, const CNodePtr &cnode2) {
  if (IsPrimitiveCNode(cnode1, prim::kPrimStridedSlice) && IsPrimitiveCNode(cnode2, prim::kPrimStridedSlice)) {
    auto slice_index1 = GetValue<int64_t>(cnode1->GetPrimalAttr(SLICE_INDEX));
    auto slice_index2 = GetValue<int64_t>(cnode2->GetPrimalAttr(SLICE_INDEX));
    return slice_index1 < slice_index2;
  }
  if (IsPrimitiveCNode(cnode1, prim::kPrimStridedSlice)) {
    return false;
  }
  return true;
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
    if (IsPrimitiveCNode(node1, prim::kPrimStridedSlice) || IsPrimitiveCNode(node2, prim::kPrimStridedSlice)) {
      return SliceSort(cnode1, cnode2);
    }
    auto prim1 = GetCNodePrimitive(cnode1);
    auto prim2 = GetCNodePrimitive(cnode2);
    if (EnableShareCell() && prim1 == nullptr && prim2 == nullptr) {
      return false;
    }
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
                  const FuncGraphPtr &root, const std::string &attr_tag) {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node);
  auto post_cnode = post_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(post_cnode);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), post_cnode->input(1), prior_node};
  auto depend_node = root->NewCNode(depend_input);
  if (!attr_tag.empty()) {
    depend_node->AddAttr(attr_tag, MakeValue<bool>(true));
  }
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
    InsertDepend(prior_node1, post_node1, manager, root, TagForSendRecDepend(prior_node1, post_node1));
    auto prior_node2 = backward_end_pair.second[LongToSize(SizeToLong(i) - stage_num + stage_id)];
    prior_node2 = GetCallBackwardEndNext(prior_node2);
    auto post_node2 = forward_start_pair.first[i];
    InsertDepend(prior_node2, post_node2, manager, root, TagForSendRecDepend(prior_node2, post_node2));
  }
  for (size_t i = LongToSize(stage_num - stage_id); i < (forward_start_pair.first.size() + 1); ++i) {
    if (!IsLastStage()) {
      auto prior_node3 = backward_start_pair.second[LongToSize(SizeToLong(i) - stage_num + stage_id)];
      auto post_node3 = forward_end_pair.first[i - 1];
      InsertDepend(prior_node3, post_node3, manager, root, TagForSendRecDepend(prior_node3, post_node3));
      auto prior_node4 = forward_end_pair.second[i - 1];
      auto post_node4 = backward_end_pair.first[LongToSize(SizeToLong(i) - stage_num + stage_id)];
      InsertDepend(prior_node4, post_node4, manager, root, TagForSendRecDepend(prior_node4, post_node4));
    }
  }
  for (size_t j = LongToSize(SizeToLong(backward_start_pair.first.size()) - stage_num + stage_id + 1);
       j < backward_start_pair.first.size(); ++j) {
    auto prior_node5 = backward_end_pair.second[j - 1];
    prior_node5 = GetCallBackwardEndNext(prior_node5);
    auto post_node5 = backward_start_pair.first[j];
    InsertDepend(prior_node5, post_node5, manager, root, TagForSendRecDepend(prior_node5, post_node5));
  }
  if (!IsLastStage()) {
    auto prior_node6 = forward_end_before_pair.second[LongToSize(stage_num - 1 - stage_id)];
    auto post_node6 = backward_start_pair.first[0];
    InsertDepend(prior_node6, post_node6, manager, root, TagForSendRecDepend(prior_node6, post_node6));
  }
}

void ReorderForParams(const PipelinePair &backward_params_pair, const PipelinePair &forward_params_pair,
                      const PipelinePair &backward_end_pair, const PipelinePair &forward_start_pair,
                      const FuncGraphPtr &root) {
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!forward_params_pair.second.empty()) {
    auto prior_node = forward_params_pair.second.back();
    auto post_node = forward_start_pair.first.front();
    InsertDepend(prior_node, post_node, manager, root);
  }
  if (!backward_params_pair.first.empty()) {
    auto prior_node2 = backward_end_pair.second.back();
    prior_node2 = GetCallBackwardEndNext(prior_node2);
    auto post_node2 = backward_params_pair.first.front();
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

void CommonDeduplicate(const std::vector<AnfNodePtr> &node_vector, std::vector<AnfNodePtr> *out_vec_begin,
                       std::vector<AnfNodePtr> *out_vec_end, const FuncGraphPtr &root, int64_t micro_max,
                       int64_t seg_max, int64_t h, bool is_train) {
  std::vector<AnfNodePtr> temp_vec;
  auto manager = root->manager();
  for (int64_t i = 0; i <= micro_max; ++i) {
    temp_vec.clear();
    if (!is_train) {
      temp_vec = node_vector;
    } else {
      for (auto &node : node_vector) {
        auto node_micro = GetMicroBatch(node);
        if (seg_max >= 1) {
          auto node_seg = GetSegment(node);
          if (node_micro == i && node_seg == h) {
            temp_vec.push_back(node);
          }
        } else {
          if (node_micro == i) {
            temp_vec.push_back(node);
          }
        }
      }
    }
    if (temp_vec.empty()) {
      MS_LOG(INFO) << "No Duplicate MicroBatch.";
      continue;
    }
    if (temp_vec.size() == 1) {
      if (seg_max >= 1) {
        MS_LOG(WARNING) << "Single element, no need to deduplicate.";
        out_vec_begin->push_back(temp_vec.front());
        out_vec_end->push_back(temp_vec.back());
      }
      continue;
    }
    std::sort(temp_vec.begin(), temp_vec.end(), CompFunc);
    for (size_t j = 0; j < temp_vec.size() - 1; ++j) {
      auto prior_node = temp_vec[j];
      prior_node = GetCallBackwardEndNext(prior_node);
      auto post_node = temp_vec[j + 1];
      InsertDepend(prior_node, post_node, manager, root);
    }
    if (!temp_vec.empty()) {
      out_vec_begin->push_back(temp_vec.front());
      out_vec_end->push_back(temp_vec.back());
    }
  }
}

PipelinePair GetForwardEndBeforePair(const PipelinePair &forward_end_pair) {
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
  return forward_end_before_pair;
}

int64_t GetMicroMax(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &forward_end) {
  int64_t micro_max = 0;
  if (forward_end.empty()) {
    MS_LOG(EXCEPTION) << "can not find the end node of pipeline, you are advised to use 'PipelineCell' to fix it.";
  } else {
    auto forward_end_cnode = forward_end.back()->cast<CNodePtr>();
    auto micro_size = forward_end_cnode->GetPrimalAttr(MICRO);
    MS_EXCEPTION_IF_NULL(micro_size);
    micro_max = GetValue<int64_t>(micro_size);
  }
  return micro_max;
}

int64_t GetSegment(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto seg_value = cnode->GetPrimalAttr(SEGMENT);
  MS_EXCEPTION_IF_NULL(seg_value);
  return GetValue<int64_t>(seg_value);
}

void BroadCastMicroBatch(const CNodePtr &node, NodeUsersMap *node_users_map, const ValuePtr &value, size_t max_depth) {
  auto node_users = (*node_users_map)[node];
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(EXCEPTION) << "Recursive call is larger than 100000.";
  }
  for (auto &node_pair : node_users) {
    auto user_node = node_pair.first->cast<CNodePtr>();
    if (user_node->HasPrimalAttr(MICRO) || IsPrimitiveCNode(user_node, prim::kPrimUpdateState)) {
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
  auto &node_user_map = manager->node_users();
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
      for (size_t i = 0; i < cnode->inputs().size(); ++i) {
        auto temp_node = GetRealKernelNode(cnode->input(i), -1, nullptr).first;
        if (!temp_node->isa<CNode>()) {
          continue;
        }
        auto temp_prim = GetCNodePrimitive(temp_node);
        if (!temp_prim || temp_prim->HasAttr(PIPELINE_END)) {
          continue;
        }
        InsertVirtualPipelineEndNode(cnode, manager, i);
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
  auto &node_users_map = manager->node_users();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(node);
    if (prim && prim->HasAttr(PARAMETER_START_SHARE_CELL)) {
      cnode->AddPrimalAttr(PARAMETER_START_SHARE_CELL, prim->GetAttr(PARAMETER_START_SHARE_CELL));
      continue;
    }
    if (prim && prim->HasAttr(PARAMETER_START)) {
      auto micro = Micro(cnode, &node_users_map, 0);
      MS_EXCEPTION_IF_NULL(micro);
      cnode->AddPrimalAttr(MICRO, micro);
      cnode->AddPrimalAttr(PARAMETER_START, micro);
      int64_t seg = 0;
      cnode->AddPrimalAttr(SEGMENT, MakeValue(seg));
    }
  }
}

void HandleMicroBatch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  auto &node_users_map = manager->node_users();
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
  int64_t slice_index = 0;
  auto all_nodes = DeepScopedGraphSearch(root->get_return());
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>() || IsPrimitiveCNode(node, prim::kPrimDepend) ||
        IsPrimitiveCNode(node, prim::kPrimZerosLike)) {
      continue;
    }
    auto prim = GetCNodePrimitive(node);
    auto cnode = node->cast<CNodePtr>();
    auto share_cell = EnableShareCell();
    if (share_cell && cnode->HasPrimalAttr(PARAMETER_START)) {
      backward_end->push_back(node);
    }
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
      if (!share_cell && cnode->HasPrimalAttr(PARAMETER_START)) {
        backward_end->push_back(node);
      }
      if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
        backward_params->push_back(node);
      }
      if (prim->HasAttr(PARAMETER_MICRO)) {
        allreduce_params->push_back(node);
      }
      continue;
    }
    // the return of cnode->HasPrimalAttr(kPrimalAttrForwardNodeName) is false.
    if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
      if (IsPrimitiveCNode(cnode, prim::kPrimStridedSlice)) {
        cnode->AddPrimalAttr(SLICE_INDEX, MakeValue(slice_index));
        slice_index += 1;
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
  std::sort((*backward_start).begin(), (*backward_start).end(), CompFuncBySegDescending);
  std::sort((*backward_end).begin(), (*backward_end).end(), CompFuncBySegDescending);
  std::sort((*forward_start).begin(), (*forward_start).end(), CompFuncBySegAscending);
  std::sort((*forward_end).begin(), (*forward_end).end(), CompFuncBySegAscending);
  std::sort((*backward_params).begin(), (*backward_params).end(), CompFunc);
  std::sort((*forward_params).begin(), (*forward_params).end(), CompFunc);
}

void CheckBorderNode(const PipelinePair &forward_start_pair, const PipelinePair &forward_end_pair,
                     const PipelinePair &backward_start_pair, const PipelinePair &backward_end_pair,
                     std::vector<int64_t> seg_micro_max) {
  auto micro_size = LongToSize(seg_micro_max[0] + 1);
  auto seg_size = LongToSize(seg_micro_max[1] + 1);
  auto total_micro_size = micro_size * seg_size;
  std::string cause = ". One possible cause is that the @lazy_inline decorator is misplaced.";
  if (forward_start_pair.first.size() != total_micro_size) {
    MS_LOG(EXCEPTION) << "forward_node's size:" << forward_start_pair.first.size()
                      << "is not equal to micro size:" << total_micro_size << cause;
  }
  if (forward_end_pair.first.size() != total_micro_size) {
    MS_LOG(EXCEPTION) << "forward_node's size:" << forward_end_pair.first.size()
                      << "is not equal to micro size:" << total_micro_size << cause;
  }
  if (backward_start_pair.first.size() != total_micro_size) {
    MS_LOG(EXCEPTION) << "backward_node's size:" << backward_start_pair.first.size()
                      << "is not equal to micro size:" << total_micro_size << cause;
  }
  if (backward_end_pair.first.size() != total_micro_size) {
    MS_LOG(EXCEPTION) << "backward_node's size:" << backward_end_pair.first.size()
                      << "is not equal to micro size:" << total_micro_size << cause;
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
  SetParameterStartForCellShare(root);
  GetBorderNode(&forward_start, &forward_end, &backward_start, &backward_end, &forward_params, &backward_params,
                &allreduce_params, root);
  int64_t micro_max = GetMicroMax(root, forward_end);
  std::vector<int64_t> seg_micro_max{micro_max, 0};
  auto backward_start_pair = Deduplicate(backward_start, root, micro_max, 0, true);
  auto backward_end_pair = Deduplicate(backward_end, root, micro_max, 0, true);
  auto forward_start_pair = Deduplicate(forward_start, root, micro_max, 0, true);
  auto forward_end_pair = Deduplicate(forward_end, root, micro_max, 0, true);
  auto forward_params_pair = Deduplicate(forward_params, root, micro_max, 0, true);
  auto backward_params_pair = Deduplicate(backward_params, root, micro_max, 0, true);
  CheckBorderNode(forward_start_pair, forward_end_pair, backward_start_pair, backward_end_pair, seg_micro_max);
  auto forward_end_before_pair = GetForwardEndBeforePair(forward_end_pair);
  auto ret_after = root->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = DeepScopedGraphSearch(ret_after);
  auto manager = root->manager();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (IsSomePrimitive(node->cast<CNodePtr>(), kNPUClearFloatStatusOpName) ||
        IsSomePrimitive(node->cast<CNodePtr>(), kNPUClearFloatStatusV2OpName)) {
      InsertDepend(node, forward_end.front(), manager, root);
      break;
    }
  }
  ReorderForForward(forward_start_pair.first, forward_end_pair.second, root);
  ReorderForBackward(forward_start_pair, forward_end_pair, backward_start_pair, backward_end_pair,
                     forward_end_before_pair, root);
  ReorderForParams(backward_params_pair, forward_params_pair, backward_end_pair, forward_start_pair, root);
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
  auto forward_start_pair = Deduplicate(forward_start, root, 0, 0, false);
  auto forward_end_pair = Deduplicate(forward_end, root, 0, 0, false);
  auto forward_params_pair = Deduplicate(forward_params, root, 0, 0, false);
  if (!forward_end.empty() && !forward_params.empty()) {
    InsertDepend(forward_params_pair.second[0], forward_end_pair.first[0], manager, root);
  }
  if (!forward_start.empty() && !forward_params.empty()) {
    InsertDepend(forward_params_pair.second[0], forward_start_pair.first[0], manager, root);
  }
}
}  // namespace parallel
}  // namespace mindspore
