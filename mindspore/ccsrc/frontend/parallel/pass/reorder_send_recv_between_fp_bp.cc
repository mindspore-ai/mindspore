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

#include "frontend/parallel/pass/reorder_send_recv_between_fp_bp.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <queue>
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kSendRecvRely = "send_recv_rely";
constexpr auto kSendRecvReorderDepend = "send_recv_reorder_depend";
void SpreadSendRecvRely(const CNodePtr &input_node, const std::string &unique_id) {
  std::queue<CNodePtr> node_queue;
  node_queue.push(input_node);
  while (!node_queue.empty()) {
    auto cnode = node_queue.front();
    node_queue.pop();
    auto cnode_inputs = cnode->inputs();
    auto spread_size = cnode_inputs.size();
    for (size_t i = 1; i < spread_size; ++i) {
      auto input = cnode_inputs[i];
      if (!IsPrimitiveCNode(input)) {
        continue;
      }
      auto input_cnode = input->cast<CNodePtr>();
      if (input_cnode->HasAttr(kSendRecvRely) &&
          GetValue<std::string>(input_cnode->GetAttr(kSendRecvRely)) == unique_id) {
        continue;
      }
      input_cnode->AddAttr(kSendRecvRely, MakeValue<std::string>(unique_id));
      node_queue.push(input_cnode);
    }
  }
}

void InsertReorderDepend(const FuncGraphPtr &graph, const CNodePtr &input_comm_cnode, const CNodePtr &user_comm_cnode,
                         const std::string &unique_id) {
  auto manager = graph->manager();
  auto input_comm_cnode_users = manager->node_users()[input_comm_cnode];
  for (const auto &input_comm_user_pair : input_comm_cnode_users) {
    if (!IsPrimitiveCNode(input_comm_user_pair.first)) {
      continue;
    }
    auto input_comm_user_cnode = input_comm_user_pair.first->cast<CNodePtr>();
    if (input_comm_user_cnode->HasAttr(SEND_REC_DEPEND) || input_comm_user_cnode->HasAttr(kSendRecvReorderDepend)) {
      continue;
    }
    if (!unique_id.empty() && input_comm_user_cnode->HasAttr(kSendRecvRely) &&
        GetValue<std::string>(input_comm_user_cnode->GetAttr(kSendRecvRely)) == unique_id) {
      continue;
    }
    std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), input_comm_cnode, user_comm_cnode};
    auto depend_node = graph->NewCNode(depend_input);
    depend_node->AddAttr(kSendRecvReorderDepend, MakeValue<bool>(true));
    depend_node->set_abstract(input_comm_cnode->abstract()->Clone());
    manager->SetEdge(input_comm_user_cnode, input_comm_user_pair.second, depend_node);
  }
}

bool is_step_in() {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return false;
  }
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() == 1) {
    return false;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->CellReuseLevel() == CellReuseLevel::kNoCellReuse;
}
}  // namespace

void ReorderSendRecvBetweenFpBp(const FuncGraphPtr &graph) {
  if (!is_step_in()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  for (const auto &cnode : origin_nodes_topological) {
    if (!IsPrimitiveCNode(cnode, prim::kPrimDepend) || !cnode->HasAttr(SEND_REC_DEPEND)) {
      continue;
    }
    // receive -> depend -> send
    //                       |
    //         -> -------- depend -> rec_user
    if (!IsPrimitiveCNode(cnode->input(INDEX_TWO), prim::kPrimSend) &&
        !IsPrimitiveCNode(cnode->input(INDEX_TWO), prim::kPrimReceive)) {
      continue;
    }
    std::string unique_id = "";
    if (IsPrimitiveCNode(cnode->input(INDEX_ONE))) {
      auto depend_node_first_input = cnode->input(INDEX_ONE)->cast<CNodePtr>();
      unique_id = depend_node_first_input->UniqueId();
      SpreadSendRecvRely(depend_node_first_input, unique_id);
    }

    auto input_comm_cnode = cnode->input(INDEX_TWO)->cast<CNodePtr>();
    CNodePtr input_comm_cnode_input_comm_cnode = nullptr;
    if (parallel::ParallelContext::GetInstance()->enable_micro_interleaved() &&
        IsPrimitiveCNode(input_comm_cnode->input(INDEX_ONE), prim::kPrimDepend)) {
      auto tmp_depend_cnode = input_comm_cnode->input(INDEX_ONE)->cast<CNodePtr>();
      tmp_depend_cnode->AddAttr(kSendRecvReorderDepend, MakeValue<bool>(true));
      if (IsPrimitiveCNode(tmp_depend_cnode->input(INDEX_TWO), prim::kPrimSend) ||
          IsPrimitiveCNode(tmp_depend_cnode->input(INDEX_TWO), prim::kPrimReceive)) {
        input_comm_cnode_input_comm_cnode = tmp_depend_cnode->input(INDEX_TWO)->cast<CNodePtr>();
      }
    }
    auto cnode_users = manager->node_users()[cnode];
    for (const auto &user_pair : cnode_users) {
      if (!IsPrimitiveCNode(user_pair.first, prim::kPrimSend) &&
          !IsPrimitiveCNode(user_pair.first, prim::kPrimReceive)) {
        continue;
      }
      auto user_comm_cnode = user_pair.first->cast<CNodePtr>();
      InsertReorderDepend(graph, input_comm_cnode, user_comm_cnode, unique_id);
      if (input_comm_cnode_input_comm_cnode != nullptr) {
        InsertReorderDepend(graph, input_comm_cnode_input_comm_cnode, user_comm_cnode, unique_id);
      }
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
