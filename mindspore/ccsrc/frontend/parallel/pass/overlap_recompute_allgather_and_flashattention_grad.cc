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

#include "frontend/parallel/pass/overlap_recompute_allgather_and_flashattention_grad.h"
#include <memory>
#include <vector>
#include <list>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <queue>
#include <utility>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace parallel {
namespace {
void AddDependForRecomputedAllGatherAndGradientReduceScatter(const FuncGraphPtr &backward_graph) {
  std::list<CNodePtr> backward_orders = backward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> backward_origin_nodes_topological(backward_orders.cbegin(), backward_orders.cend());
  auto manager = backward_graph->manager();
  std::unordered_map<std::string, CNodePtr> recomputed_ag_map;
  std::unordered_map<std::string, CNodePtr> grad_rs_map;
  for (const auto &recomputed_allgather : backward_origin_nodes_topological) {
    if (!IsPrimitiveCNode(recomputed_allgather, prim::kPrimAllGather)) {
      continue;
    }
    if (!recomputed_allgather->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (!recomputed_allgather->HasPrimalAttr(kPrimalAttrUniqueId)) {
      continue;
    }
    auto unique_id = GetValue<std::string>(recomputed_allgather->GetPrimalAttr(kPrimalAttrUniqueId));
    recomputed_ag_map[unique_id] = recomputed_allgather;
  }
  for (const auto &grad_reducescatter : backward_origin_nodes_topological) {
    if (!IsPrimitiveCNode(grad_reducescatter, prim::kPrimReduceScatter)) {
      continue;
    }
    if (!grad_reducescatter->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      continue;
    }
    auto unique_id = GetValue<std::string>(grad_reducescatter->GetPrimalAttr(kPrimalAttrForwardUniqueId));
    grad_rs_map[unique_id] = grad_reducescatter;
  }
  for (const auto &ag_pair : recomputed_ag_map) {
    if (grad_rs_map.count(ag_pair.first) == 0) {
      continue;
    }
    auto recomputed_ag = ag_pair.second;
    auto grad_rs = grad_rs_map[ag_pair.first];
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), grad_rs->input(kIndex1), recomputed_ag};
    auto depend_node = backward_graph->NewCNode(depend_inputs);
    depend_node->set_abstract(grad_rs->input(kIndex1)->abstract()->Clone());
    (void)manager->SetEdge(grad_rs, kIndex1, depend_node);
    depend_node->AddAttr("recompute_ag_grad_rs_depend", MakeValue(true));
  }
}

AnfNodePtr GetDependRealNode(const AnfNodePtr &depend_rely_node) {
  AnfNodePtr real_depend_rely_node = nullptr;
  if (IsPrimitiveCNode(depend_rely_node, prim::kPrimMakeTuple)) {
    auto make_tuple_node = depend_rely_node->cast<CNodePtr>();
    for (size_t i = 1; i < make_tuple_node->size(); ++i) {
      auto make_tuple_input = GetInputNodeWithFilter(make_tuple_node->input(i), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem) || IsPrimitiveCNode(cnode, prim::kPrimConcat) ||
                      IsPrimitiveCNode(cnode, prim::kPrimReshape) || IsPrimitiveCNode(cnode, prim::kPrimSplit) ||
                      IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimTranspose);
        return std::make_pair(filter, 1);
      });
      if (IsPrimitiveCNode(make_tuple_input, prim::kPrimFlashAttentionScoreGrad)) {
        real_depend_rely_node = make_tuple_node->input(i);
        break;
      }
    }
  } else {
    real_depend_rely_node = depend_rely_node;
  }
  return real_depend_rely_node;
}

void OverlapRecomputeAGAndFlashAttentionGrad(const FuncGraphPtr &backward_graph) {
  std::list<CNodePtr> backward_orders = backward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> backward_origin_nodes_topological(backward_orders.cbegin(), backward_orders.cend());
  auto manager = backward_graph->manager();
  auto node_users = manager->node_users();
  for (const auto &recomputed_allgather : backward_origin_nodes_topological) {
    if (!IsPrimitiveCNode(recomputed_allgather, prim::kPrimAllGather)) {
      continue;
    }
    if (!recomputed_allgather->HasAttr(kAttrDuplicated)) {
      continue;
    }
    auto ag_input = recomputed_allgather->input(kIndex1);
    auto pre_cnode = GetInputNodeWithFilter(ag_input, [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                    IsPrimitiveCNode(cnode, prim::kPrimReshape);
      return std::make_pair(filter, 1);
    });
    if (!IsPrimitiveCNode(pre_cnode, prim::kPrimDepend)) {
      continue;
    }
    auto depend_cnode = pre_cnode->cast<CNodePtr>();
    auto depend_rely_node = depend_cnode->input(kIndex2);
    auto real_depend_rely_node = GetDependRealNode(depend_rely_node);
    if (!real_depend_rely_node) {
      continue;
    }
    auto new_rely_node = GetInputNodeWithFilter(real_depend_rely_node, [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem) || IsPrimitiveCNode(cnode, prim::kPrimConcat) ||
                    IsPrimitiveCNode(cnode, prim::kPrimReshape) || IsPrimitiveCNode(cnode, prim::kPrimSplit) ||
                    IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimTranspose);
      return std::make_pair(filter, 1);
    });
    if (!IsPrimitiveCNode(new_rely_node, prim::kPrimFlashAttentionScoreGrad)) {
      continue;
    }

    auto fa_grad = new_rely_node->cast<CNodePtr>();
    auto new_depend_rely = GetInputNodeWithFilter(fa_grad->input(kIndex4), [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimReshape);
      return std::make_pair(filter, 1);
    });

    (void)manager->SetEdge(depend_cnode, kIndex2, new_depend_rely);
    depend_cnode->AddAttr(kAttrRecomputeCommDepend, MakeValue(true));
    if (!IsPrimitiveCNode(depend_rely_node) || depend_rely_node->cast<CNodePtr>()->size() < SIZE_TWO ||
        depend_rely_node == new_rely_node) {
      MS_LOG(INFO) << "Depend_rely_node: " << depend_rely_node->fullname_with_scope()
                   << ", new_rely_node: " << new_rely_node->fullname_with_scope();
      continue;
    }
    std::vector<AnfNodePtr> depend_inputs{
      NewValueNode(prim::kPrimDepend), real_depend_rely_node->cast<CNodePtr>()->input(kIndex1), recomputed_allgather};
    auto depend_node = backward_graph->NewCNode(depend_inputs);
    depend_node->set_abstract(real_depend_rely_node->cast<CNodePtr>()->input(kIndex1)->abstract()->Clone());
    (void)manager->SetEdge(real_depend_rely_node, kIndex1, depend_node);
    depend_node->AddAttr("recompute_ag_fa_grad_depend1", MakeValue(true));

    // fa_grad -> allgather_users
    for (const auto &ag_user : node_users.at(recomputed_allgather)) {
      std::vector<AnfNodePtr> depend_inputs1{NewValueNode(prim::kPrimDepend), recomputed_allgather, fa_grad};
      auto depend_node1 = backward_graph->NewCNode(depend_inputs1);
      depend_node1->set_abstract(recomputed_allgather->abstract()->Clone());
      (void)manager->SetEdge(ag_user.first, ag_user.second, depend_node1);
      depend_node1->AddAttr("recompute_ag_fa_grad_depend2", MakeValue(true));
    }
  }
}

}  // namespace

void OverlapRecomputeAllGatherAndFlashAttentionGrad(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto soc_version = ms_context->ascend_soc_version();
  if (soc_version != "ascend910" && soc_version != "ascend910b" && soc_version != "ascend910c") {
    return;
  }
  const auto cell_reuse = ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (!cell_reuse) {
    MS_LOG(WARNING)
      << "Currently, duplicated allgather overlap with flashattention grad only support in lazy_line mode.";
    return;
  }
  auto is_enable = ms_context->get_param<bool>(MS_CTX_RECOMPUTE_ALLGATHER_OVERLAP_FAGRAD);
  if (!is_enable) {
    return;
  }
  auto manager = graph->manager();
  FuncGraphPtr backward_graph = graph;
  for (const auto &each_graph : manager->func_graphs()) {
    if (IsCellReuseForwardGraph(each_graph)) {
      auto forward_graph = each_graph;
      // need to using the inlined backward_graph
      backward_graph = GetCellReuseBackwardGraph(forward_graph);
      if (backward_graph == nullptr) {
        MS_LOG(WARNING)
          << "Failed to find backward cell reuse graph, skip pass 'overlap_gradmatmul_and_gradallreduce'.";
        return;
      }
      break;
    }
  }
  OverlapRecomputeAGAndFlashAttentionGrad(backward_graph);
  AddDependForRecomputedAllGatherAndGradientReduceScatter(backward_graph);
}
}  // namespace parallel
}  // namespace mindspore
