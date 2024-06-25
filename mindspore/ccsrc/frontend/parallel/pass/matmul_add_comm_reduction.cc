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

#include "frontend/parallel/pass/matmul_add_comm_reduction.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <utility>
#include "include/common/utils/utils.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr size_t kCommReductionValidCommOpsNum = 2;
constexpr auto MATMUL_ADD_COMM_BEGIN = "matmul_add_comm_begin";
constexpr auto MATMUL_ADD_COMM_END = "matmul_add_comm_end";
constexpr auto MATMUL_ADD_COMM_MUL = "matmul_add_comm_mul";
constexpr const char MATMUL_ADD_COMM_REDUCTION[] = "matmul_add_comm_reduction";

bool IsSubRankList(const RankList &child_list, const RankList &parent_list) {
  for (auto &child : child_list) {
    if (std::find(parent_list.begin(), parent_list.end(), child) == parent_list.end()) {
      return false;
    }
  }
  return true;
}

bool IsPrimitiveAttrValid(const PrimitivePtr &prim, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(prim);
  return !prim->HasAttr(attr_name) || !GetValue<bool>(prim->GetAttr(attr_name));
}

bool IsAddNodeValid(const AnfNodePtr &add_node, const AnfNodePtr &comm_node) {
  OperatorInfoPtr add_distribute_operator = add_node->user_data<OperatorInfo>();
  if (add_distribute_operator == nullptr) {
    return false;
  }
  TensorInfo node_add_tensor_in = add_distribute_operator->inputs_tensor_info()[LongToSize(1)];
  TensorLayout node_add_tensor_layout = node_add_tensor_in.tensor_layout();
  const auto node_add_rank_list = node_add_tensor_layout.InferRepeatedGroup();

  auto comm_prim = GetCNodePrimitive(comm_node);
  if (!comm_prim->HasAttr(GROUP)) {
    return false;
  }
  auto comm_group = GetValue<std::string>(comm_prim->GetAttr(GROUP));
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto comm_rank_list = g_device_manager->FindRankListByHashName(comm_group);
  return IsSubRankList(comm_rank_list, node_add_rank_list);
}

bool IsPrimitiveLinear(const AnfNodePtr &anode) {
  MS_EXCEPTION_IF_NULL(anode);
  if (IsPrimitiveCNode(anode, prim::kPrimReduceAll) || IsPrimitiveCNode(anode, prim::kPrimReduceAny) ||
      IsPrimitiveCNode(anode, prim::kPrimReduceMean) || IsPrimitiveCNode(anode, prim::kPrimReduceMax) ||
      IsPrimitiveCNode(anode, prim::kPrimReduceMin) || IsPrimitiveCNode(anode, prim::kPrimReduceProd) ||
      IsPrimitiveCNode(anode, prim::kPrimReduceSum) || IsPrimitiveCNode(anode, prim::kPrimSquareSumV1)) {
    return false;
  }
  return true;
}

AnfNodePtr FindPullDownNode(const AnfNodePtr &anode) {
  auto pre_node = GetInputNodeWithFilter(anode, [&](const AnfNodePtr &cur_anode) {
    auto cur_cnode = cur_anode->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cur_cnode);
    if (prim == nullptr) {
      return std::make_pair(false, LongToSize(0));
    }
    auto cur_node_input_list = cur_cnode->inputs();
    for (size_t i = 1; i < cur_node_input_list.size(); ++i) {
      auto cur_input_node = cur_node_input_list[i];
      // find first non Tensor CNode
      if (IsValueNode<tensor::Tensor>(cur_input_node)) {
        continue;
      }
      auto input_prim = GetCNodePrimitive(cur_input_node);
      if (input_prim == nullptr) {
        return std::make_pair(false, i);
      }
      // cur prim must in ALLREDUCE_PULL_DOWN_WHITE_LIST and input_prim is not marked or marked false
      bool filter = (ALLREDUCE_PULL_DOWN_WHITE_LIST.find(prim->name()) != ALLREDUCE_PULL_DOWN_WHITE_LIST.end() ||
                     prim->name() == MATMUL || prim->name() == BATCH_MATMUL) &&
                    IsPrimitiveAttrValid(input_prim, MATMUL_ADD_COMM_BEGIN);
      return std::make_pair(filter, i);
    }
    return std::make_pair(false, LongToSize(1));
  });
  return pre_node;
}

void FindAllValidAddNode(const FuncGraphPtr &graph, HashMap<AnfNodePtr, std::vector<AnfNodePtr>> *pull_down_node_map) {
  std::list<CNodePtr> graph_orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
  for (const auto &node : origin_nodes_topological) {
    // add node
    auto prim = GetCNodePrimitive(node);
    if (prim == nullptr || prim->name() != ADD || IsPrimitiveAttrValid(prim, MATMUL_ADD_COMM_END)) {
      continue;
    }
    auto input_nodes = node->inputs();
    for (size_t i = 1; i < input_nodes.size(); ++i) {
      auto input_node = input_nodes[i];
      if (!IsPrimitiveLinear(input_node)) {
        continue;
      }
      auto comm_node = FindPullDownNode(input_node);
      if (comm_node == nullptr) {
        MS_LOG(INFO) << "For matmul add comm reduction, can not find valid comm node, node is "
                     << input_node->DebugString();
        continue;
      }
      if ((!IsPrimitiveCNode(comm_node, prim::kPrimAllReduce) &&
           !IsPrimitiveCNode(comm_node, prim::kPrimReduceScatter))) {
        MS_LOG(INFO) << "For matmul comm reduction, comm node is not allreduce or reduce scatter, node is "
                     << comm_node->DebugString();
        continue;
      }

      auto comm_cnode = comm_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(comm_node);
      auto pre_prim = GetCNodePrimitive(comm_cnode->input(1));
      if (pre_prim == nullptr || IsPrimitiveAttrValid(pre_prim, MATMUL_ADD_COMM_BEGIN)) {
        MS_LOG(INFO) << "For matmul comm reduction,  cannot find matmul/batch matmul node, "
                     << "skip cur node: " << input_node->DebugString();
        continue;
      }
      (*pull_down_node_map)[node].push_back(comm_node);
      MS_LOG(INFO) << "For matmul comm reduction, find one side with matmul-allreduce structure, add node is: "
                   << node->DebugString() << " comm node is: " << comm_node->DebugString();
    }
  }
}

AnfNodePtr FindBiasAdd(const AnfNodePtr &comm_node, const AnfNodePtr &add_node_input) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto add_node = GetInputNodeWithFilter(add_node_input, [&](const AnfNodePtr &anode) {
    auto prim = GetCNodePrimitive(anode);
    if (prim == nullptr) {
      return std::make_pair(false, 0);
    }
    // find add node, current ops must lie in ALLREDUCE_PULL_DOWN_WHITE_LIST, cannot be add node or equal to comm node
    bool filter = (ALLREDUCE_PULL_DOWN_WHITE_LIST.find(prim->name()) != ALLREDUCE_PULL_DOWN_WHITE_LIST.end() ||
                   prim->name() == MATMUL || prim->name() == BATCH_MATMUL) &&
                  prim->name() != ADD && anode != comm_node;
    return std::make_pair(filter, 1);
  });
  return add_node;
}

void HandleNodeBiasAdd(const AnfNodePtr &comm_node, const AnfNodePtr &add_node_input) {
  MS_EXCEPTION_IF_NULL(comm_node);
  MS_EXCEPTION_IF_NULL(add_node_input);
  auto comm_prim = GetCNodePrimitive(comm_node);
  MS_EXCEPTION_IF_NULL(comm_prim);
  if (!comm_prim->HasAttr(GROUP)) {
    MS_LOG(INFO) << "For matmul comm reduction, cur prim has not attr " << GROUP
                 << ", skip it, node is: " << comm_node->DebugString();
    return;
  }
  auto comm_group = GetValue<std::string>(comm_prim->GetAttr(GROUP));
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto comm_rank_list = g_device_manager->FindRankListByHashName(comm_group);
  double rank_size = 1.0 / comm_rank_list.size();

  auto add_node = FindBiasAdd(comm_node, add_node_input);
  if (add_node == nullptr || !IsPrimitiveCNode(add_node, prim::kPrimAdd)) {
    MS_LOG(INFO) << "For matmul comm reduction, cannot find bias add node, find node is: " << add_node->DebugString()
                 << " start node is " << add_node_input->DebugString();
    return;
  }
  if (!IsAddNodeValid(add_node, comm_node)) {
    MS_LOG(INFO) << "For matmul comm reduction, strategy of add node mismatched, skip it, add node is: "
                 << add_node->DebugString();
    return;
  }
  auto add_cnode = add_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add_cnode);
  // find load node for bias parameter
  auto bias_side_start_node = add_cnode->input(2);
  auto bias_node = GetInputNodeWithFilter(bias_side_start_node, [&](const AnfNodePtr &anode) {
    auto prim = GetCNodePrimitive(anode);
    if (prim == nullptr) {
      return std::make_pair(false, 0);
    }
    bool filter = ALLREDUCE_PULL_DOWN_WHITE_LIST.find(prim->name()) != ALLREDUCE_PULL_DOWN_WHITE_LIST.end();
    return std::make_pair(filter, 1);
  });
  if (bias_node == nullptr || !IsPrimitiveCNode(bias_node, prim::kPrimLoad)) {
    MS_LOG(INFO) << "For comm reduction, cannot find load op for bias parameter along current add node, please "
                    "check whether it exists, cur add node is: "
                 << add_node->DebugString();
    return;
  }
  // insert mul node
  auto bias_node_abstract = bias_node->abstract();
  MS_EXCEPTION_IF_NULL(bias_node_abstract);
  auto bias_dtype = bias_node_abstract->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(bias_dtype);
  auto bias_dtype_ele = bias_dtype->element();
  MS_EXCEPTION_IF_NULL(bias_dtype_ele);
  mindspore::tensor::TensorPtr tensor_ptr =
    std::make_shared<mindspore::tensor::Tensor>(rank_size, bias_dtype_ele->GetType());
  auto const_node = NewValueNode(MakeValue(tensor_ptr));
  const_node->set_abstract(const_node->value()->ToAbstract());

  auto mul_prim = NewValueNode(prim::kPrimMul);
  auto cur_prim = GetValueNode<PrimitivePtr>(mul_prim);
  MS_EXCEPTION_IF_NULL(cur_prim);
  (void)cur_prim->AddAttr(MATMUL_ADD_COMM_MUL, MakeValue(true));
  AnfNodePtrList mul_node_inputs = {mul_prim, bias_node, const_node};
  auto fg = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto mul_node = fg->NewCNode(mul_node_inputs);
  mul_node->set_abstract(bias_node->abstract()->Clone());

  MS_EXCEPTION_IF_NULL(fg);
  auto manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(bias_node, mul_node);
  MS_LOG(INFO) << "for comm reduction, insert new mul node after parameter node";
}

void HandleNodePullUp(const AnfNodePtr &add_node, const std::vector<AnfNodePtr> &comm_node_list,
                      HashMap<AnfNodePtr, AnfNodePtr> *comm_node_map) {
  for (size_t index = 0; index < comm_node_list.size(); ++index) {
    // Node pull down
    // Node After AllReduce pull up
    auto each_node = comm_node_list[index];
    auto each_cnode = each_node->cast<CNodePtr>();
    auto pre_node = each_cnode->input(1);
    auto pre_prim = GetCNodePrimitive(pre_node);
    if (pre_prim == nullptr || IsPrimitiveAttrValid(pre_prim, MATMUL_ADD_COMM_BEGIN)) {
      MS_LOG(INFO) << "For comm reduction, its pre node does not marked or marked false, skip it.";
      continue;
    }
    auto graph = each_node->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto add_cnode = add_node->cast<CNodePtr>();
    HandleNodeBiasAdd(each_node, add_cnode->input(index + 1));
    (void)manager->Replace(each_node, pre_node);
    MS_LOG(INFO) << "For comm reduction, pull up node next to comm node, node is: " << pre_node->DebugString();
    if ((*comm_node_map).find(add_node) == (*comm_node_map).end()) {
      (*comm_node_map)[add_node] = each_node;
    }
  }
}

void HandleNodePullDown(const AnfNodePtr &add_node, const AnfNodePtr &comm_node) {
  auto comm_cnode = comm_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(comm_cnode);
  AnfNodePtrList new_comm_node_inputs = {comm_cnode->input(0), add_node};
  auto graph = add_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto new_comm_node = graph->NewCNode(new_comm_node_inputs);
  new_comm_node->set_abstract(comm_node->abstract());
  auto prim = GetCNodePrimitive(new_comm_node);
  (void)prim->AddAttr(MATMUL_ADD_COMM_REDUCTION, MakeValue(true));

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(add_node, new_comm_node);
  MS_LOG(INFO) << "For comm reduction, pull down comm node, node is: " << new_comm_node->DebugString();
}

void HandleAddNode(const HashMap<AnfNodePtr, std::vector<AnfNodePtr>> &pull_down_node_map) {
  HashMap<AnfNodePtr, AnfNodePtr> comm_node_map;
  for (auto &each_pull_down_node : pull_down_node_map) {
    if (each_pull_down_node.second.size() < kCommReductionValidCommOpsNum) {
      MS_LOG(INFO) << "For comm reduction, cur node cannot find match structure, skip it. current node is "
                   << each_pull_down_node.first->DebugString();
      continue;
    }
    // Handle node pull up
    HandleNodePullUp(each_pull_down_node.first, each_pull_down_node.second, &comm_node_map);
    // Handle node pull down
    HandleNodePullDown(each_pull_down_node.first, comm_node_map[each_pull_down_node.first]);
  }
}

}  // namespace

// For Structure as following:
//  MatMul/BatchMatMul -> AllReduce -> ... -> X -> Add, and MatMul/BatchMatMul -> AllReduce -> ... -> Y -> Add
// Change it to MatMul/BatchMatMul -> ... -> X -> Add -> AllReduce and MatMul/BatchMatMul -> ... -> Y -> Add ->
// AllReduce thus it can reduce a communication op.
bool MatmulAddCommReduction(const FuncGraphPtr &graph, const opt::OptimizerPtr &) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // assume no change to graph
  bool changes = false;
  HashMap<AnfNodePtr, std::vector<AnfNodePtr>> pull_down_node_map;
  // candidate node to pull down
  for (const auto &each_graph : manager->func_graphs()) {
    FindAllValidAddNode(each_graph, &pull_down_node_map);
  }
  // Node Pull up
  HandleAddNode(pull_down_node_map);
  return changes;
}
}  // namespace parallel
}  // namespace mindspore
