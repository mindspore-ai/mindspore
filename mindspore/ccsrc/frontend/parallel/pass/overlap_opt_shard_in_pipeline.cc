/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/overlap_opt_shard_in_pipeline.h"
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include "mindspore/core/ops/core_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/comm_manager.h"

namespace mindspore {
namespace parallel {
namespace {
inline bool is_allgather_comm_ops(const AnfNodePtr &node) {
  static const std::vector<PrimitivePtr> kAllGatherOpsPrim = {prim::kPrimMicroStepAllGather,
                                                              prim::kPrimMiniStepAllGather, prim::kPrimAllGather};
  for (const auto &prim : kAllGatherOpsPrim) {
    if (IsPrimitiveCNode(node, prim)) {
      auto allgather_instance_name = GetCNodePrimitive(node->cast<CNodePtr>())->instance_name();
      if (allgather_instance_name.find(parallel::PARALLEL_OPTIMIZER) == std::string::npos) {
        return false;
      }
      return true;
    }
  }
  return false;
}

inline bool is_first_receive(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    auto recv_node = node->cast<CNodePtr>();
    if (recv_node->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      return false;
    }
    auto micro = GetValue<int64_t>(recv_node->GetPrimalAttr(parallel::MICRO));
    if (micro != 0 || recv_node->HasPrimalAttr(parallel::PIPELINE_PARAM)) {
      return false;
    }
    return true;
  }
  return false;
}
}  // namespace

void OverlapOptShardInPipeline(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!parallel::IsAutoParallelCareGraph(graph) ||
      parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() <= 1 ||
      parallel::ParallelContext::GetInstance()->grad_accumulation_shard()) {
    return;
  }
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  CNodePtr first_receive_cnode = nullptr;
  for (auto &node : origin_nodes_topological) {
    if (is_first_receive((node))) {
      first_receive_cnode = node->cast<CNodePtr>();
      first_receive_cnode->AddAttr(parallel::FIRST_RECEIVE, MakeValue(True));
    }
  }
  if (first_receive_cnode == nullptr) {
    return;
  }
  auto recv_users = manager->node_users()[first_receive_cnode];
  if (recv_users.empty()) {
    return;
  }

  std::vector<CNodePtr> opt_shard_allgather_list;
  for (auto &node : origin_nodes_topological) {
    MS_EXCEPTION_IF_NULL(node);
    if (!is_allgather_comm_ops(node)) {
      continue;
    }
    auto cnode_allgather = node->cast<CNodePtr>();
    opt_shard_allgather_list.push_back(cnode_allgather);
    auto allgather_prim = GetCNodePrimitive(cnode_allgather);
    auto group_name = GetValue<std::string>(allgather_prim->GetAttr(parallel::GROUP));
    if (group_name.find("parallel_optimizer") != std::string::npos) {
      continue;
    }
    auto rank_ids = parallel::g_device_manager->FindRankListByHashName(group_name);
    if (rank_ids.empty()) {
      continue;
    }
    auto dev_list = parallel::g_device_manager->CreateDeviceListByRankList(rank_ids);
    auto new_group_name = group_name + "_parallel_optimizer";
    parallel::Group cur_device_list;
    (void)parallel::g_device_manager->CreateGroup(new_group_name, dev_list, &cur_device_list);
    (void)allgather_prim->AddAttr(parallel::GROUP, MakeValue<std::string>(new_group_name));
  }
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy(opt_shard_allgather_list.begin(), opt_shard_allgather_list.end(),
                  std::back_inserter(make_tuple_inputs));
  for (auto &user_set : recv_users) {
    if (!IsPrimitiveCNode(user_set.first)) {
      continue;
    }
    auto recv_user = user_set.first->cast<CNodePtr>();
    auto recv_user_index = user_set.second;
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), first_receive_cnode,
                                          graph->NewCNode(make_tuple_inputs)};
    auto depend_node = graph->NewCNode(depend_inputs);
    depend_node->set_abstract(first_receive_cnode->abstract()->Clone());
    depend_node->AddAttr("RecAllGatherDepend", MakeValue(True));
    manager->SetEdge(recv_user, recv_user_index, depend_node);
  }
}
}  // namespace parallel
}  // namespace mindspore
