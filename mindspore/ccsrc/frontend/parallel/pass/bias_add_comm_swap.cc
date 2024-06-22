/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/bias_add_comm_swap.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <utility>
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr const char BIAS_ADD_COMM_SWAP[] = "bias_add_comm_swap";

bool IsSubRankList(const RankList &child_list, const RankList &parent_list) {
  for (auto &child : child_list) {
    if (std::find(parent_list.begin(), parent_list.end(), child) == parent_list.end()) {
      return false;
    }
  }
  return true;
}
bool IsAddNodeValid(const CNodePtr &add_node, const AnfNodePtr &comm_node) {
  OperatorInfoPtr add_distribute_operator = add_node->user_data<OperatorInfo>();
  if (add_distribute_operator == nullptr) {
    return false;
  }
  TensorInfo node_add_tensor_in = add_distribute_operator->inputs_tensor_info()[LongToSize(1)];
  TensorLayout node_add_tensor_layout = node_add_tensor_in.tensor_layout();
  auto node_add_rank_list = node_add_tensor_layout.InferRepeatedGroup();

  auto comm_prim = GetCNodePrimitive(comm_node);
  if (!comm_prim->HasAttr(GROUP)) {
    return false;
  }
  auto comm_group = GetValue<std::string>(comm_prim->GetAttr(GROUP));
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto comm_rank_list = g_device_manager->FindRankListByHashName(comm_group);
  return IsSubRankList(comm_rank_list, node_add_rank_list);
}

// find matmul node
AnfNodePtr FindMatMulNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto matmul_node = GetInputNodeWithFilter(node, [&](const CNodePtr &cnode) {
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      return std::make_pair(false, 0);
    }
    bool filter = (ALLREDUCE_PULL_DOWN_WHITE_LIST.find(prim->name()) != ALLREDUCE_PULL_DOWN_WHITE_LIST.end()) ||
                  (IsPrimitiveCNode(cnode, prim::kPrimAllReduce) || IsPrimitiveCNode(cnode, prim::kPrimReduceScatter));
    return std::make_pair(filter, 1);
  });
  return matmul_node;
}

// find valid allreduce/reduce_scatter node
AnfNodePtr FindValidCommNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto comm_node = GetInputNodeWithFilter(node, [&](const AnfNodePtr &anode) {
    auto prim = GetCNodePrimitive(anode);
    if (prim == nullptr) {
      return std::make_pair(false, 0);
    }
    bool filter = ALLREDUCE_PULL_DOWN_WHITE_LIST.find(prim->name()) != ALLREDUCE_PULL_DOWN_WHITE_LIST.end();
    return std::make_pair(filter, 1);
  });
  if (comm_node == nullptr ||
      (!IsPrimitiveCNode(comm_node, prim::kPrimAllReduce) && !IsPrimitiveCNode(comm_node, prim::kPrimReduceScatter))) {
    return nullptr;
  }
  auto matmul_node = FindMatMulNode(comm_node);
  if (matmul_node == nullptr || !IsPrimitiveCNode(matmul_node, prim::kPrimMatMul)) {
    MS_LOG(WARNING) << "For bias_add_comm_swap, cannot find matmul node from comm node, comm node is: "
                    << comm_node->DebugString();
    return nullptr;
  }
  return comm_node;
}

void FindAllValidAddNode(const FuncGraphPtr &graph, HashMap<CNodePtr, AnfNodePtr> *add_node_map) {
  std::list<CNodePtr> graph_orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
  for (const auto &node : origin_nodes_topological) {
    if (!IsPrimitiveCNode(node, prim::kPrimAdd)) {
      continue;
    }

    auto comm_node = FindValidCommNode(node);
    if (comm_node == nullptr) {
      MS_LOG(INFO) << "For bias_add_comm_swap, cannot find valid comm node, cur node is " << node->DebugString();
      continue;
    }
    if (!IsAddNodeValid(node, comm_node)) {
      MS_LOG(INFO) << "For bias_add_comm_swap, strategy of add node and comm node are not equal, cur node is "
                   << node->DebugString() << " comm node is " << comm_node->DebugString();
      continue;
    }
    (*add_node_map)[node] = comm_node;
  }
}

void HandleNodePullUp(const AnfNodePtr &comm_node, const CNodePtr &add_node) {
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // handle matmul node, connect it to next node of reduce_scatter/allreduce
  auto comm_cnode = comm_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(comm_cnode);
  auto comm_node_input = comm_cnode->input(1);
  MS_EXCEPTION_IF_NULL(comm_node_input);
  (void)manager->Replace(comm_node, comm_node_input);
}

void changeNodeShape(const CNodePtr &add_node, size_t rank_size) {
  MS_EXCEPTION_IF_NULL(add_node);
  auto add_node_abstract = add_node->abstract();
  MS_EXCEPTION_IF_NULL(add_node_abstract);
  auto add_node_shape_ptr = add_node_abstract->GetShape();
  MS_EXCEPTION_IF_NULL(add_node_shape_ptr);
  auto add_node_shape = add_node_shape_ptr->GetShapeVector();
  if (add_node_shape.size() < 1) {
    return;
  }
  add_node_shape[0] = add_node_shape[0] * SizeToLong(rank_size);
  auto new_shape_item = std::make_shared<abstract::Shape>(add_node_shape);
  add_node_abstract->set_shape(new_shape_item);
  add_node->set_abstract(add_node_abstract);
}

bool HandleNodeBiasAdd(const AnfNodePtr &comm_node, const CNodePtr &add_node) {
  auto comm_prim = GetCNodePrimitive(comm_node);
  MS_EXCEPTION_IF_NULL(comm_prim);
  if (!comm_prim->HasAttr(GROUP)) {
    MS_LOG(WARNING) << "For matmul comm reduction, cur prim has not attr " << GROUP
                    << ", skip it, node is: " << comm_node->DebugString();
    return false;
  }
  auto comm_group = GetValue<std::string>(comm_prim->GetAttr(GROUP));
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto comm_rank_list = g_device_manager->FindRankListByHashName(comm_group);
  double rank_size = 1.0 / comm_rank_list.size();

  // change node shape
  if (IsPrimitiveCNode(comm_node, prim::kPrimReduceScatter)) {
    changeNodeShape(add_node, comm_rank_list.size());
  }

  auto bias_side_start_node = add_node->input(2);
  MS_EXCEPTION_IF_NULL(bias_side_start_node);
  auto bias_node = GetInputNodeWithFilter(bias_side_start_node, [&](const AnfNodePtr &anode) {
    auto prim = GetCNodePrimitive(anode);
    if (prim == nullptr) {
      return std::make_pair(false, 0);
    }
    bool filter = ALLREDUCE_PULL_DOWN_WHITE_LIST.find(prim->name()) != ALLREDUCE_PULL_DOWN_WHITE_LIST.end();
    return std::make_pair(filter, 1);
  });
  if (bias_node == nullptr || !IsPrimitiveCNode(bias_node, prim::kPrimLoad)) {
    MS_LOG(WARNING) << "From cur add node, cannot find Load op for bias parameter, please check whether it exists,"
                    << "  cur node is: " << add_node->DebugString();
    return true;
  }
  auto load_node_shape_ptr = bias_node->Shape();
  if (load_node_shape_ptr == nullptr) {
    return false;
  }
  auto load_node_shape = load_node_shape_ptr->GetShapeVector();
  if (load_node_shape.size() > 1) {
    MS_LOG(WARNING) << "for bias add comm swap, bias shape can not larger than 1, but got " << load_node_shape;
    return false;
  }
  auto bias_node_abstract = bias_node->abstract();
  MS_EXCEPTION_IF_NULL(bias_node_abstract);
  const auto bias_dtype = bias_node_abstract->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(bias_dtype);
  mindspore::tensor::TensorPtr tensor_ptr =
    std::make_shared<mindspore::tensor::Tensor>(rank_size, bias_dtype->element()->GetType());
  auto const_node = NewValueNode(MakeValue(tensor_ptr));
  const_node->set_abstract(const_node->value()->ToAbstract());
  AnfNodePtrList mul_node_inputs = {NewValueNode(prim::kPrimMul), bias_node, const_node};

  auto fg = comm_node->func_graph();
  auto mul_node = fg->NewCNode(mul_node_inputs);
  mul_node->set_abstract(bias_node->abstract()->Clone());
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(bias_node, mul_node);
  MS_LOG(INFO) << "For bias add comm swap, insert mul node after bias parameter, node is: " << bias_node->DebugString();
  return true;
}

void HandleNodePullDown(const AnfNodePtr &comm_node, const CNodePtr &add_node) {
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  AnfNodePtrList new_comm_node_inputs = {comm_node->cast<CNodePtr>()->input(0), add_node};
  auto new_comm_node = graph->NewCNode(new_comm_node_inputs);
  new_comm_node->set_abstract(comm_node->abstract());
  auto prim = GetCNodePrimitive(new_comm_node);
  (void)prim->AddAttr(BIAS_ADD_COMM_SWAP, MakeValue(true));
  (void)manager->Replace(add_node, new_comm_node);
}

void HandleAddNode(HashMap<CNodePtr, AnfNodePtr> *add_node_map) {
  for (auto node_pair : (*add_node_map)) {
    auto add_node = node_pair.first;
    auto comm_node = node_pair.second;
    auto is_bias_node_valid = HandleNodeBiasAdd(comm_node, add_node);
    if (!is_bias_node_valid) {
      return;
    }
    HandleNodePullUp(comm_node, add_node);
    // pull down comm node, change add node user's input to allreduce
    HandleNodePullDown(comm_node, add_node);
  }
}
}  // namespace

void BiasAddCommSwap(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_BIAS_ADD_COMM_SWAP)) {
    return;
  }

  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  HashMap<CNodePtr, AnfNodePtr> add_node_map;
  for (auto &each_graph : manager->func_graphs()) {
    FindAllValidAddNode(each_graph, &add_node_map);
  }
  // pull up add node, pull down allreduce/reduce_scatter node
  HandleAddNode(&add_node_map);
}
}  // namespace parallel
}  // namespace mindspore
