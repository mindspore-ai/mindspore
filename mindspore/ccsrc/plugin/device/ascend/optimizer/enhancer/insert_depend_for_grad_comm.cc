/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <map>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_grad_comm.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kGradientsFlag = "Gradients";
struct AnfNodeCmp {
  bool operator()(const AnfNodePtr &node1, const AnfNodePtr &node2) { return node1->UniqueId() < node2->UniqueId(); }
};

std::vector<AnfNodePtr> GetInputNodeSkipComm(const CNodePtr &node) {
  std::queue<AnfNodePtr> anf_queue;
  std::vector<AnfNodePtr> visited;
  for (size_t i = 1; i < node->size(); ++i) {
    anf_queue.push(node->input(i));
    visited.push_back(node->input(i));
  }

  std::vector<AnfNodePtr> res;
  while (!anf_queue.empty()) {
    auto queue_front = anf_queue.front();
    anf_queue.pop();
    if (!(IsPrimitiveCNode(queue_front, prim::kPrimTupleGetItem) ||
          IsPrimitiveCNode(queue_front, prim::kPrimAllReduce) ||
          IsPrimitiveCNode(queue_front, prim::kPrimReduceScatter))) {
      res.push_back(queue_front);
      continue;
    }
    auto cnode_queue_end = queue_front->cast<CNodePtr>();
    for (size_t i = 1; i < cnode_queue_end->size(); ++i) {
      if (std::find(visited.begin(), visited.end(), cnode_queue_end->input(i)) != visited.end()) {
        continue;
      }
      if (IsPrimitiveCNode(queue_front, prim::kPrimTupleGetItem) && i == 2) {
        continue;
      }
      anf_queue.push(cnode_queue_end->input(i));
      visited.push_back(cnode_queue_end->input(i));
    }
  }
  return res;
}

bool IsSomePrimitiveList(const CNodePtr &cnode, const std::set<string> &check_list) {
  if (!cnode) {
    return false;
  }
  ValueNodePtr anf_node = cnode->input(0)->cast<ValueNodePtr>();
  if (!anf_node) {
    return false;
  }
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  if (!prim) {
    return false;
  }
  return std::any_of(check_list.begin(), check_list.end(), [prim](const string &in) { return prim->name() == in; });
}

std::vector<std::pair<AnfNodePtr, int>> GetOutputNodesSkipComm(const FuncGraphManagerPtr &manager,
                                                               const AnfNodePtr &node) {
  std::vector<std::pair<AnfNodePtr, int>> res;
  std::queue<std::pair<AnfNodePtr, int>> anf_queue;
  std::vector<AnfNodePtr> visited;
  auto node_users_map = manager->node_users();
  for (const auto &node_pair : node_users_map[node]) {
    anf_queue.push(node_pair);
    visited.push_back(node_pair.first);
  }
  while (!anf_queue.empty()) {
    auto queue_front = anf_queue.front();
    anf_queue.pop();
    if (!IsSomePrimitiveList(queue_front.first->cast<CNodePtr>(),
                             {prim::kPrimTupleGetItem->name(), prim::kPrimAllReduce->name(),
                              prim::kPrimReduceScatter->name(), prim::kPrimMakeTuple->name()})) {
      res.push_back(queue_front);
      continue;
    }
    for (const auto &node_pair : node_users_map[queue_front.first]) {
      if (std::find(visited.begin(), visited.end(), node_pair.first) != visited.end()) {
        continue;
      }
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimDepend) && node_pair.second == 2) {
        continue;
      }
      anf_queue.push(node_pair);
      visited.push_back(node_pair.first);
    }
  }
  return res;
}

bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return IsBpropNode(node->cast<CNodePtr>()->input(1));
  }
  return node->fullname_with_scope().find(kGradientsFlag) == 0;
}

bool IsDxMatMul(const CNodePtr &matmul_node) {
  std::queue<AnfNodePtr> cnode_queue;
  std::vector<AnfNodePtr> visited;
  for (size_t i = 1; i < matmul_node->size(); ++i) {
    cnode_queue.push(matmul_node->input(i));
    visited.push_back(matmul_node->input(i));
  }

  std::vector<AnfNodePtr> res;
  while (!cnode_queue.empty()) {
    auto queue_front = cnode_queue.front();
    cnode_queue.pop();
    if (!IsSomePrimitiveList(queue_front->cast<CNodePtr>(),
                             {prim::kPrimTransData->name(), prim::kPrimLoad->name(), prim::kPrimDepend->name(),
                              prim::kPrimTupleGetItem->name(), prim::kPrimTensorMove->name(), prim::kPrimConcat->name(),
                              prim::kPrimConcatD->name()})) {
      res.push_back(queue_front);
      continue;
    }
    auto cnode_queue_end = queue_front->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode_queue_end, prim::kPrimTupleGetItem) &&
        IsPrimitiveCNode(cnode_queue_end->input(1), prim::kPrimMakeTuple)) {
      auto make_tuple_cnode = cnode_queue_end->input(1)->cast<CNodePtr>();
      ValuePtr tuple_index_value = GetValueNode(cnode_queue_end->input(2));
      MS_EXCEPTION_IF_NULL(tuple_index_value);
      if (!tuple_index_value->isa<Int64Imm>()) {
        MS_LOG(EXCEPTION) << "The index of tuple getitem is not int64";
      }
      auto tupleget_item_index = tuple_index_value->cast<Int64ImmPtr>()->value();
      auto make_tuple_input_index = tupleget_item_index + 1;
      auto real_input_node = make_tuple_cnode->input(make_tuple_input_index);
      cnode_queue.push(real_input_node);
      visited.push_back(real_input_node);
      continue;
    }
    if (std::find(visited.begin(), visited.end(), cnode_queue_end->input(1)) != visited.end()) {
      continue;
    }
    cnode_queue.push(cnode_queue_end->input(1));
    visited.push_back(cnode_queue_end->input(1));
  }
  for (const auto &node : res) {
    if (node->isa<Parameter>()) {
      return true;
    }
    if (IsPrimitiveCNode(node, prim::kPrimAllGather)) {
      return true;
    }
  }
  return false;
}

void GetCandidateNodes(const std::set<AnfNodePtr, AnfNodeCmp> &input_set, std::set<AnfNodePtr> *visited,
                       std::vector<CNodePtr> *candidate_nodes) {
  std::queue<CNodePtr> cnode_queue;
  for (const auto &anf_node : input_set) {
    if (!anf_node->isa<CNode>()) {
      continue;
    }
    cnode_queue.push(anf_node->cast<CNodePtr>());
  }
  while (!cnode_queue.empty()) {
    auto queue_front = cnode_queue.front();
    cnode_queue.pop();
    for (size_t i = 1; i < queue_front->size(); ++i) {
      if (std::find((*visited).begin(), (*visited).end(), queue_front->input(i)) != (*visited).end()) {
        continue;
      }
      visited->insert(queue_front->input(i));
      auto prim = GetCNodePrimitive(queue_front->input(i));
      if (!prim || prim->HasAttr("in_strategy")) {
        continue;
      }
      auto input_cnode = queue_front->input(i)->cast<CNodePtr>();
      cnode_queue.push(input_cnode);
      if (!IsPrimitiveCNode(input_cnode, prim::kPrimMatMul) || !IsBpropNode(input_cnode)) {
        continue;
      }
      // Check whether a input is weight
      if (!IsDxMatMul(input_cnode)) {
        continue;
      }
      candidate_nodes->push_back(input_cnode);
    }
  }
}

CNodePtr RelyNode(const NodeUsersMap &node_users_map, const std::set<AnfNodePtr, AnfNodeCmp> &input_set,
                  const std::vector<AnfNodePtr> &node_list) {
  std::set<AnfNodePtr> visited(input_set.begin(), input_set.end());
  std::vector<CNodePtr> candidate_nodes;
  GetCandidateNodes(input_set, &visited, &candidate_nodes);
  // Find the last dx matmul
  std::sort(candidate_nodes.begin(), candidate_nodes.end(), [&node_list](auto &cnode1, auto &cnode2) {
    return static_cast<int>(std::find(node_list.begin(), node_list.end(), cnode1) - node_list.begin()) >
           static_cast<int>(std::find(node_list.begin(), node_list.end(), cnode2) - node_list.begin());
  });

  for (const auto &rely_node : candidate_nodes) {
    std::queue<CNodePtr> new_cnode_queue;
    new_cnode_queue.push(rely_node);
    std::set<AnfNodePtr, AnfNodeCmp> new_visited;
    new_visited.insert(rely_node);
    while (!new_cnode_queue.empty() && new_visited.size() < 500) {
      auto queue_front = new_cnode_queue.front();
      new_cnode_queue.pop();
      if (IsPrimitiveCNode(queue_front, prim::kPrimMatMul) && IsBpropNode(queue_front) && IsDxMatMul(queue_front) &&
          std::find(visited.begin(), visited.end(), queue_front) == visited.end()) {
        return queue_front;
      }
      if (node_users_map.count(queue_front) == 0) {
        continue;
      }
      for (const auto &node_pair : node_users_map.at(queue_front)) {
        if (std::find(new_visited.begin(), new_visited.end(), node_pair.first) != new_visited.end() ||
            !IsPrimitiveCNode(node_pair.first)) {
          continue;
        }
        new_visited.insert(node_pair.first);
        new_cnode_queue.push(node_pair.first->cast<CNodePtr>());
      }
    }
  }
  return nullptr;
}

std::map<int64_t, std::vector<CNodePtr>> GradCommNode(const std::vector<AnfNodePtr> &node_list) {
  std::map<int64_t, std::vector<CNodePtr>> grad_comm_node;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimAllReduce) && !IsPrimitiveCNode(node, prim::kPrimReduceScatter)) {
      continue;
    }
    auto grad_comm_cnode = node->cast<CNodePtr>();
    if (!(common::AnfAlgo::HasNodeAttr(kAttrFusion, grad_comm_cnode) &&
          common::AnfAlgo::GetNodeAttr<int64_t>(grad_comm_cnode, kAttrFusion) > 0)) {
      continue;
    }
    auto fusion_id = common::AnfAlgo::GetNodeAttr<int64_t>(grad_comm_cnode, kAttrFusion);
    grad_comm_node[fusion_id].push_back(grad_comm_cnode);
  }
  return grad_comm_node;
}

bool IsStepIn() {
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    return false;
  }
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    return false;
  }
  auto is_enable_grad_comm_opt = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_GRAD_COMM_OPT);
  if (!is_enable_grad_comm_opt) {
    return false;
  }
  return true;
}
}  // namespace

bool InsertDependForGradComm::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!IsStepIn()) {
    return false;
  }
  auto manager = graph->manager();
  auto node_users_map = manager->node_users();
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  auto grad_comm_node = GradCommNode(node_list);
  std::vector<std::set<AnfNodePtr, AnfNodeCmp>> current_node_outputs_list;
  std::vector<CNodePtr> rely_nodes_list;
  auto iter = grad_comm_node.rbegin();
  for (int64_t i = SizeToInt(grad_comm_node.size()) - 1; i >= 0; --i) {
    auto current_node_list = iter->second;
    std::set<AnfNodePtr, AnfNodeCmp> current_node_outputs;
    for (const auto &comm_node : current_node_list) {
      auto comm_node_users = GetOutputNodesSkipComm(manager, comm_node);
      for (const auto &user_pair : comm_node_users) {
        current_node_outputs.insert(user_pair.first);
      }
    }
    std::set<AnfNodePtr, AnfNodeCmp> current_node_inputs;
    for (const auto &comm_node : current_node_list) {
      auto comm_node_inputs = GetInputNodeSkipComm(comm_node);
      for (const auto &comm_input : comm_node_inputs) {
        current_node_inputs.insert(comm_input);
      }
    }
    current_node_outputs_list.push_back(current_node_outputs);
    auto rely_node = RelyNode(node_users_map, current_node_inputs, node_list);
    rely_nodes_list.push_back(rely_node);
    if (!rely_node) {
      continue;
    }
    // current_node_inputs -> depend -> rely_node
    std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
    std::copy(current_node_inputs.begin(), current_node_inputs.end(), std::back_inserter(make_tuple_inputs));
    auto current_input_nodes_tuple = graph->NewCNode(make_tuple_inputs);
    std::vector<AnfNodePtr> depend_node_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                                  rely_node->input(1), current_input_nodes_tuple};
    auto depend_node = graph->NewCNode(depend_node_inputs);
    depend_node->AddAttr("grad_comm_depend1", MakeValue(i + 1));
    manager->SetEdge(rely_node, 1, depend_node);
    ++iter;
    changed = True;
  }

  // next_rely_node -> depend -> current_node_outputs
  for (size_t i = 0; i + 1 < rely_nodes_list.size(); ++i) {
    auto rely_node = rely_nodes_list[i + 1];
    if (!rely_node) {
      continue;
    }
    auto current_node_outputs = current_node_outputs_list[i];
    for (const auto &current_node_output : current_node_outputs) {
      if (!IsPrimitiveCNode(current_node_output)) {
        continue;
      }
      auto output_cnode = current_node_output->cast<CNodePtr>();
      std::vector<AnfNodePtr> depend_node2_inputs = {
        NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), output_cnode->input(1), rely_node};
      auto depend_node2 = graph->NewCNode(depend_node2_inputs);
      depend_node2->AddAttr("grad_comm_depend2", MakeValue(i + 1));
      manager->SetEdge(output_cnode, 1, depend_node2);
    }
  }

  // insert depend for comm_input_i -> comm_input_i1
  auto f_iter = grad_comm_node.rbegin();
  for (int64_t i = 0; i < SizeToInt(grad_comm_node.size()) - 1; ++i) {
    auto current_node_list = f_iter->second;
    auto next_node_list = (++f_iter)->second;
    std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
    std::copy(current_node_list.begin(), current_node_list.end(), std::back_inserter(make_tuple_inputs));
    auto current_comm_nodes_tuple = graph->NewCNode(make_tuple_inputs);
    for (const auto &next_comm_node : next_node_list) {
      if (!IsPrimitiveCNode(next_comm_node)) {
        continue;
      }
      auto next_comm_cnode = next_comm_node->cast<CNodePtr>();
      std::vector<AnfNodePtr> depend_node_inputs = {
        NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), next_comm_cnode->input(1),
        current_comm_nodes_tuple};
      auto depend_node = graph->NewCNode(depend_node_inputs);
      depend_node->AddAttr("grad_comm_depend3", MakeValue(true));
      manager->SetEdge(next_comm_cnode, 1, depend_node);
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
