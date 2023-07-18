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

#include <memory>
#include <list>
#include <set>
#include <queue>
#include <algorithm>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/graph_util/grad_accumulation_utils.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "ir/value.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "utils/parallel_node_check.h"

namespace mindspore {
namespace parallel {
constexpr char GRAD_ACCU_NUM[] = "grad_accu_num";
constexpr char GRAD_ACCU_FORWARD_BEGIN[] = "grad_accu_forward_begin";
constexpr char GRAD_ACCU_FORWARD_END[] = "grad_accu_forward_end";
constexpr char GRAD_ACCU_BACKWARD_END[] = "grad_accu_backward_end";
constexpr char FIRST_PARAMETER_CNODE[] = "first_parameter_cnode";
void TagMicroBatchStart(const FuncGraphManagerPtr &manager, const std::vector<AnfNodePtr> &all_nodes) {
  auto node_users_map = manager->node_users();
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimStridedSlice)) {
      continue;
    }
    auto slice_cnode = node->cast<CNodePtr>();
    auto slice_prim = GetCNodePrimitive(slice_cnode);
    if (!slice_prim->HasAttr(GRAD_ACCU_NUM)) {
      continue;
    }
    auto accu_step = GetValue<int64_t>(slice_prim->GetAttr(GRAD_ACCU_NUM));
    ParallelContext::GetInstance()->set_grad_accumulation_step(accu_step);
    auto value = GetValueNode(slice_cnode->input(2));
    MS_EXCEPTION_IF_NULL(value);
    auto tuple = GetValue<std::vector<int64_t>>(value);
    auto input_tmp = GetNodeShape(slice_cnode->input(1));
    auto input_shape = input_tmp.at(0);
    int64_t micro = tuple.at(0) * accu_step / input_shape.at(0);
    slice_cnode->AddPrimalAttr(MICRO, MakeValue(micro));
    slice_cnode->AddPrimalAttr(GRAD_ACCU_FORWARD_BEGIN, MakeValue(micro));
    MS_LOG(INFO) << "Find grad accumulation begin node.";
    BroadCastMicroBatch(slice_cnode, &node_users_map, MakeValue(micro), 0);
  }
}

void TagMicroBatchEnd(const FuncGraphManagerPtr &manager, const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node)) {
      continue;
    }
    auto end_cnode = node->cast<CNodePtr>();
    auto end_prim = GetCNodePrimitive(end_cnode);
    if (!end_prim->HasAttr(FORWARD_END)) {
      continue;
    }
    if (ParallelContext::GetInstance()->grad_accumulation_step() > 1 && !end_cnode->HasPrimalAttr(MICRO)) {
      MS_LOG(EXCEPTION) << "Cannot find micro attribute for forward_end nodes";
    }
    for (size_t i = 0; i < end_cnode->inputs().size(); ++i) {
      auto temp_node = GetRealKernelNode(end_cnode->input(i), -1, nullptr).first;
      if (!temp_node->isa<CNode>()) {
        continue;
      }
      auto temp_prim = GetCNodePrimitive(temp_node);
      if (!temp_prim || temp_prim->HasAttr(FORWARD_END)) {
        continue;
      }
      InsertVirtualPipelineEndNode(end_cnode, manager, i, GRAD_ACCU_FORWARD_END);
    }
  }
}

ValuePtr SearchPreNodeMicro(const CNodePtr &cnode) {
  if (cnode->HasPrimalAttr(MICRO)) {
    return cnode->GetPrimalAttr(MICRO);
  }
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (!cnode->input(i)->isa<CNode>()) {
      continue;
    }
    return SearchPreNodeMicro(cnode->input(i)->cast<CNodePtr>());
  }
  return nullptr;
}

void TagMicroBatchBpEndInCellShare(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  auto node_users_map = manager->node_users();
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (const auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode->input(0), prim::kPrimTupleGetItem)) {
      continue;
    }

    auto tuple_getitem_cnode = cnode->input(0)->cast<CNodePtr>();
    auto tuple_getitem_cnode_input = tuple_getitem_cnode->input(1)->cast<CNodePtr>();
    if (!tuple_getitem_cnode_input || !IsValueNode<FuncGraph>(tuple_getitem_cnode_input->input(0))) {
      continue;
    }
    auto reuse_graph = GetValueNode<FuncGraphPtr>(tuple_getitem_cnode_input->input(0));
    if (!reuse_graph->has_flag("no_inline")) {
      continue;
    }
    MS_LOG(INFO) << "Find bp call func node:" << node->DebugString();

    auto micro = SearchPreNodeMicro(cnode);
    if (!micro) {
      MS_LOG(EXCEPTION) << "Cannot find micro info in cell share for node:" << node->DebugString();
    }
    const auto &users = node_users_map[node];
    for (const auto &user : users) {
      const auto &cuser = user.first->cast<CNodePtr>();
      if (!cuser) {
        continue;
      }
      if (IsPrimitiveCNode(cuser, prim::kPrimTupleGetItem) && IsValidNode(cuser, root->get_return(), node_users_map)) {
        cuser->AddPrimalAttr(GRAD_ACCU_BACKWARD_END, micro);
        break;
      }
    }
  }
}

void TagMicroBatchBpEndPrim(const FuncGraphPtr &root) {
  FuncGraphPtr parallel_care_graph = nullptr;
  for (auto &fg : root->manager()->func_graphs()) {
    for (auto &node : fg->nodes()) {
      if (IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
        parallel_care_graph = fg;
        break;
      }
    }
  }
  if (!parallel_care_graph) {
    MS_LOG(EXCEPTION) << "Cannot find parallel care graph with VirtualDataset";
  }
  bool is_found = false;
  auto orders = parallel_care_graph->GetOrderedCnodes();
  for (auto node = orders.cbegin(); node != orders.cend(); ++node) {
    auto cnode = (*node)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    if (!prim || !IsParallelConsiderCNode(cnode) ||
        IsSomePrimitiveList(cnode, {prim::kPrimTupleGetItem->name(), prim::kPrimMakeTuple->name()})) {
      continue;
    }
    for (size_t i = 1; i < cnode->size(); ++i) {
      std::pair<AnfNodePtr, bool> param_node_pair = FindParameter(cnode->input(i), parallel_care_graph);
      if (param_node_pair.first) {
        (void)prim->AddAttr(FIRST_PARAMETER_CNODE, MakeValue(0));
        is_found = true;
        break;
      }
    }
    if (is_found) {
      break;
    }
  }
}

void TagMicroBatchBpEnd(const FuncGraphPtr &root) {
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    if (!prim->HasAttr(FIRST_PARAMETER_CNODE)) {
      continue;
    }
    auto micro = SearchPreNodeMicro(cnode->cast<CNodePtr>());
    if (!micro) {
      MS_LOG(EXCEPTION) << "Cannot find micro info for node:" << node->DebugString();
    }
    cnode->AddPrimalAttr(GRAD_ACCU_BACKWARD_END, micro);
  }
}

void ExtractMicroBatchBorderNodes(const FuncGraphPtr &root,
                                  std::unordered_map<int64_t, std::vector<CNodePtr>> *forward_start,
                                  std::unordered_map<int64_t, std::vector<CNodePtr>> *backward_end) {
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    bool is_bp_node = cnode->HasPrimalAttr(kPrimalAttrForwardNodeName);
    if (!is_bp_node && cnode->HasPrimalAttr(GRAD_ACCU_FORWARD_BEGIN)) {
      auto accu_forward_begin_micro = GetValue<int64_t>(cnode->GetPrimalAttr(GRAD_ACCU_FORWARD_BEGIN));
      (*forward_start)[accu_forward_begin_micro].push_back(cnode);
    }
    if ((is_bp_node || IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) &&
        cnode->HasPrimalAttr(GRAD_ACCU_BACKWARD_END)) {
      auto accu_backward_end_micro = GetValue<int64_t>(cnode->GetPrimalAttr(GRAD_ACCU_BACKWARD_END));
      (*backward_end)[accu_backward_end_micro].push_back(cnode);
    }
  }
}

void ReorderGradAccumulation(const FuncGraphPtr &root,
                             const std::unordered_map<int64_t, std::vector<CNodePtr>> &forward_start,
                             const std::unordered_map<int64_t, std::vector<CNodePtr>> &backward_end) {
  if (forward_start.empty() || backward_end.empty()) {
    MS_LOG(EXCEPTION) << "Cannot find grad_accumulation border node.";
  }
  auto manager = root->manager();
  for (int64_t micro = 0; micro < ParallelContext::GetInstance()->grad_accumulation_step() - 1; ++micro) {
    if (forward_start.find(micro + 1) == forward_start.end()) {
      MS_LOG(EXCEPTION) << "Micro " << micro + 1 << " cannot find forward_start nodes.";
    }
    if (backward_end.find(micro) == backward_end.end()) {
      MS_LOG(EXCEPTION) << "Micro " << micro << " cannot find backward_end nodes.";
    }
    // backward_end -> depend -> next_forward_start
    std::vector<AnfNodePtr> backward_end_inputs{NewValueNode(prim::kPrimMakeTuple)};
    std::copy(backward_end.at(micro).begin(), backward_end.at(micro).end(), std::back_inserter(backward_end_inputs));
    auto backward_end_make_tuple_cnode = root->NewCNode(backward_end_inputs);
    for (const auto &forward_start_node : forward_start.at(micro + 1)) {
      std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), forward_start_node->input(1),
                                            backward_end_make_tuple_cnode};
      auto depend_node = root->NewCNode(depend_inputs);
      depend_node->AddAttr("grad_accu_reorder2", MakeValue(micro));
      manager->SetEdge(forward_start_node, 1, depend_node);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
