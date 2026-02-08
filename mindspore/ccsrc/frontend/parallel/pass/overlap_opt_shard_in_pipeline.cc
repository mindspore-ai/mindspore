/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/framework_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/utils/utils.h"
#include "include/utils/comm_manager.h"
#include "ir/graph_utils.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace parallel {
bool is_allgather_comm_ops(const AnfNodePtr &node) {
  static const std::vector<PrimitivePtr> kAllGatherOpsPrim = {prim::kPrimMicroStepAllGather,
                                                              prim::kPrimMiniStepAllGather, prim::kPrimAllGather};

  for (const auto &prim : kAllGatherOpsPrim) {
    if (IsPrimitiveCNode(node, prim)) {
      MS_EXCEPTION_IF_NULL(node);
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto cnode_primitive = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(cnode_primitive);
      auto allgather_instance_name = cnode_primitive->instance_name();
      if (allgather_instance_name.find(parallel::PARALLEL_OPTIMIZER) == std::string::npos) {
        return false;
      }
      return true;
    }
  }
  return false;
}

bool is_first_receive(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
    auto recv_node = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(recv_node);
    if (recv_node->HasPrimalAttr(PIPELINE_PARAM)) {
      return false;
    }
    if (recv_node->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      return false;
    }
    if (recv_node->HasPrimalAttr(parallel::CHUNK)) {
      auto chunk = GetValue<int64_t>(recv_node->GetPrimalAttr(parallel::CHUNK));
      if (chunk != 0) {
        return false;
      }
    }
    if (recv_node->HasPrimalAttr(parallel::MICRO)) {
      auto micro = GetValue<int64_t>(recv_node->GetPrimalAttr(parallel::MICRO));
      if (micro != 0) {
        return false;
      }
    } else {
      return false;
    }
    return true;
  }
  return false;
}

void OverlapOptShardInPipeline(const FuncGraphPtr &graph) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  static const bool is_enable_ge = context->GetBackend() == "GE";
  if (is_enable_ge) {
    return;
  }
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  if (!IsTraining(graph->manager())) {
    MS_LOG(INFO) << "Skip overlap in Evaluation.";
    return;
  }
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!parallel::IsAutoParallelCareGraph(graph) ||
      parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() <= 1 ||
      parallel::ParallelContext::GetInstance()->grad_accumulation_shard()) {
    return;
  }
  if (parallel::ParallelContext::GetInstance()->enable_fold_pipeline()) {
    return;
  }
  circle_handler::SetAttrToDepend(graph);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  std::vector<CNodePtr> first_receive_cnode_list;
  std::copy_if(origin_nodes_topological.begin(), origin_nodes_topological.end(),
               std::back_inserter(first_receive_cnode_list), [](const auto &node) { return is_first_receive(node); });
  std::vector<CNodePtr> opt_shard_allgather_list;
  std::vector<AbstractBasePtr> maketuple_abs_inputs;
  std::copy_if(origin_nodes_topological.begin(), origin_nodes_topological.end(),
               std::back_inserter(opt_shard_allgather_list),
               [](const auto &node) { return is_allgather_comm_ops(node); });
  std::transform(opt_shard_allgather_list.begin(), opt_shard_allgather_list.end(),
                 std::back_inserter(maketuple_abs_inputs),
                 [](const auto &cnode_allgather) { return cnode_allgather->abstract(); });
  if (opt_shard_allgather_list.empty() || first_receive_cnode_list.empty()) {
    return;
  }
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy(opt_shard_allgather_list.begin(), opt_shard_allgather_list.end(),
                  std::back_inserter(make_tuple_inputs));
  auto make_tuple_cnode = graph->NewCNode(make_tuple_inputs);
  make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(maketuple_abs_inputs));
  for (const auto &first_receive_cnode : first_receive_cnode_list) {
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), first_receive_cnode->input(kIndex1),
                                          make_tuple_cnode};
    auto depend_node = graph->NewCNode(depend_inputs);
    depend_node->set_abstract(first_receive_cnode->input(kIndex1)->abstract()->Clone());
    depend_node->AddAttr("RecAllGatherDepend", MakeValue(True));
    (void)manager->SetEdge(first_receive_cnode, kIndex1, depend_node);
  }
  circle_handler::DetectAndRevertGraphCircle(graph, manager, "OverlapOptShardInPipeline");
}

static std::vector<CNodePtr> GetOptShardReduceScatter(const std::vector<AnfNodePtr> &all_nodes) {
  std::vector<CNodePtr> reduce_scatters;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimReduceScatter)) {
      continue;
    }
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    auto instance_name = prim->instance_name();
    if (instance_name.find(kAttrNeedAllGather) != std::string::npos) {
      (void)reduce_scatters.emplace_back(node->cast<CNodePtr>());
    }
  }
  return reduce_scatters;
}

void OverlapOptShardGradInPipeline(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (parallel::ParallelContext::GetInstance()->grad_accumulation_shard() ||
      parallel::ParallelContext::GetInstance()->zero3()) {
    return;
  }
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  const auto disable_ge_kernel = IsDisableGeKernel();
  if (disable_ge_kernel) {
    return;
  }
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto stage_num = g_device_manager->stage_num();
  if (stage_num <= 1) {
    return;
  }
  circle_handler::SetAttrToDepend(graph);
  auto ret_after = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret_after);
  auto all_nodes = TopoSort(ret_after, SuccDeeperSimple);
  std::vector<CNodePtr> sends;
  int64_t micro_size = 1;
  CNodePtr last_send = nullptr;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      continue;
    }
    sends.emplace_back(cnode);
    auto micro_attr = cnode->GetPrimalAttr(MICRO);
    if (micro_attr == nullptr) {
      continue;
    }
    auto micro = GetValue<int64_t>(micro_attr) + 1;
    if (micro > micro_size) {
      micro_size = micro;
      last_send = cnode;
    }
  }
  if (last_send != nullptr) {
    auto opt_shard_rs = GetOptShardReduceScatter(all_nodes);
    if (opt_shard_rs.empty()) {
      return;
    }
    auto manager = graph->manager();
    for (auto rs : opt_shard_rs) {
      std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), rs->input(kIndex1), last_send};
      auto depend = graph->NewCNode(depend_input);
      depend->AddPrimalAttr(PP_OPT_SHARD_CONTROL, MakeValue(1));
      depend->set_abstract(rs->input(kIndex1)->abstract());
      manager->SetEdge(rs, 1, depend);
    }
  }
  circle_handler::DetectAndRevertGraphCircle(graph, graph->manager(), "OverlapOptShardGradInPipeline");
}
}  // namespace parallel
}  // namespace mindspore
