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

#include "frontend/parallel/pass/fusion_mc2_op.h"
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <utility>
#include <memory>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
namespace {
// return true if rank_ids is a continuous 8p group
bool IsSingleNodeCommGroup(const std::vector<uint32_t> &rank_ids) {
  auto group_size = rank_ids.size();
  if (group_size != 8) {
    return false;
  }
  auto rank_ids_cpy = rank_ids;
  std::sort(rank_ids_cpy.begin(), rank_ids_cpy.end());
  if (rank_ids_cpy[0] % group_size != 0) {
    return false;
  }
  for (size_t i = 1; i < group_size; ++i) {
    if (rank_ids_cpy[i - 1] + 1 != rank_ids_cpy[i]) {
      return false;
    }
  }
  return true;
}

bool IsNodesDTypeSameAndValid(const std::vector<AnfNodePtr> &nodes, const std::vector<TypeId> &valid_types) {
  if (nodes.empty()) {
    return true;
  }
  std::vector<TypeId> types;
  for (const auto &node : nodes) {
    (void)types.emplace_back(common::AnfAlgo::GetOutputInferDataType(node, kIndex0));
  }
  if (std::find(valid_types.begin(), valid_types.end(), types[0]) == valid_types.end()) {
    return false;
  }
  auto type0 = types[0];
  return std::all_of(types.begin() + 1, types.end(), [&type0](TypeId type) { return type == type0; });
}

// MatMul -> ReduceScatter
bool MatmulReduceScatterPatternFilter(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimReduceScatter)) {
    return true;
  }
  auto reduce_scatter_cnode = node->cast<CNodePtr>();
  auto matmul_node = reduce_scatter_cnode->input(kIndex1);
  if (!IsPrimitiveCNode(matmul_node, prim::kPrimMatMul)) {
    return true;
  }
  auto matmul_cnode = matmul_node->cast<CNodePtr>();
  if (matmul_node->func_graph() != reduce_scatter_cnode->func_graph()) {
    return true;
  }
  auto matmul_cnode_users = matmul_cnode->func_graph()->manager()->node_users()[matmul_cnode];
  if (matmul_cnode_users.size() != 1) {
    return true;
  }
  return false;
}

void MatmulReduceScatterFusion(const FuncGraphPtr &func_graph) {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto todo =
    DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, MatmulReduceScatterPatternFilter);
  MS_LOG(INFO) << "For graph " << func_graph->ToString() << ", the candidate nodes size is " << todo.size();
  for (const auto &reduce_scatter_node : todo) {
    if (MatmulReduceScatterPatternFilter(reduce_scatter_node) || reduce_scatter_node->func_graph() != func_graph) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(reduce_scatter_node);
    auto reduce_scatter_cnode = reduce_scatter_node->cast<CNodePtr>();
    auto reduce_scatter_prim = GetCNodePrimitive(reduce_scatter_cnode);
    auto rank_list = GetValue<std::vector<uint32_t>>(reduce_scatter_prim->GetAttr(kAttrRankList));
    // Only support 8p comm group currently.
    if (!IsSingleNodeCommGroup(rank_list)) {
      continue;
    }
    auto matmul_cnode = reduce_scatter_cnode->input(kIndex1)->cast<CNodePtr>();
    auto matmul_prim = GetCNodePrimitive(matmul_cnode);
    const std::vector<TypeId> valid_types = {kFloat16->type_id(), kBFloat16->type_id()};
    if (!IsNodesDTypeSameAndValid({matmul_cnode->input(kIndex1), matmul_cnode->input(kIndex2)}, valid_types)) {
      continue;
    }
    auto matmul_reduce_scatter_prim = prim::kPrimMatmulReduceScatter->Clone();
    matmul_reduce_scatter_prim->AddAttr(kAttrGroup, reduce_scatter_prim->GetAttr(kAttrGroup));
    matmul_reduce_scatter_prim->AddAttr(kAttrRankSize, reduce_scatter_prim->GetAttr(kAttrRankSize));
    matmul_reduce_scatter_prim->AddAttr(kAttrReduceOp, reduce_scatter_prim->GetAttr(kAttrOp));
    matmul_reduce_scatter_prim->AddAttr(kAttrCommTurn, MakeValue<int64_t>(0));  // default value: 0
    matmul_reduce_scatter_prim->AddAttr(kAttrIsTransA, matmul_prim->GetAttr(kAttrTransposeX1));
    matmul_reduce_scatter_prim->AddAttr(kAttrIsTransB, matmul_prim->GetAttr(kAttrTransposeX2));

    auto matmul_reduce_scatter_cnode = func_graph->NewCNode(
      {NewValueNode(matmul_reduce_scatter_prim), matmul_cnode->input(kIndex1), matmul_cnode->input(kIndex2)});
    matmul_reduce_scatter_cnode->set_abstract(reduce_scatter_cnode->abstract());

    // Replace graph
    auto prev_cnode = matmul_cnode->input(kIndex1);
    manager->SetEdge(matmul_reduce_scatter_cnode, kIndex1, matmul_cnode->input(kIndex1));
    manager->SetEdge(matmul_reduce_scatter_cnode, kIndex2, matmul_cnode->input(kIndex2));
    auto next_cnode_users = manager->node_users()[reduce_scatter_cnode];
    for (const auto &next_cnode_pair : next_cnode_users) {
      manager->SetEdge(next_cnode_pair.first, next_cnode_pair.second, matmul_reduce_scatter_cnode);
    }
    MS_LOG(INFO) << "Fusion " << matmul_cnode->fullname_with_scope() << " and "
                 << reduce_scatter_cnode->fullname_with_scope() << " succeed,";
  }
  return;
}

// AllGather->MatMul
bool AllGatherMatmulPatternFilter(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMatMul)) {
    return true;
  }
  auto matmul_cnode = node->cast<CNodePtr>();
  auto matmul_prim = GetCNodePrimitive(matmul_cnode);
  MS_EXCEPTION_IF_NULL(matmul_prim);
  auto is_trans_x1 = GetValue<bool>(matmul_prim->GetAttr(kAttrTransposeX1));
  if (is_trans_x1) {
    return true;
  }
  auto all_gather_node = matmul_cnode->input(kIndex1);
  if (!IsPrimitiveCNode(all_gather_node, prim::kPrimAllGather)) {
    return true;
  }
  auto all_gather_cnode = all_gather_node->cast<CNodePtr>();
  auto all_gather_cnode_users = all_gather_cnode->func_graph()->manager()->node_users()[all_gather_cnode];
  return std::any_of(all_gather_cnode_users.begin(), all_gather_cnode_users.end(),
                     [&all_gather_cnode](const std::pair<AnfNodePtr, int> &all_gather_cnode_user_pair) {
                       return all_gather_cnode_user_pair.first->func_graph() != all_gather_cnode->func_graph();
                     });
}

void AllGatherMatmulFusion(const FuncGraphPtr &func_graph) {
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto todo = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, AllGatherMatmulPatternFilter);
  MS_LOG(INFO) << "For graph " << func_graph->ToString() << ", the candidate nodes size is " << todo.size();
  std::set<AnfNodePtr> fused_all_gather_set;
  for (const auto &matmul_node : todo) {
    if (AllGatherMatmulPatternFilter(matmul_node) || matmul_node->func_graph() != func_graph) {
      continue;
    }
    auto matmul_cnode = matmul_node->cast<CNodePtr>();
    auto matmul_prim = GetCNodePrimitive(matmul_cnode);
    auto all_gather_cnode = matmul_cnode->input(kIndex1)->cast<CNodePtr>();
    if (fused_all_gather_set.find(all_gather_cnode) != fused_all_gather_set.end()) {
      continue;
    }
    fused_all_gather_set.insert(all_gather_cnode);
    auto all_gather_prim = GetCNodePrimitive(all_gather_cnode);
    auto rank_list = GetValue<std::vector<uint32_t>>(all_gather_prim->GetAttr(kAttrRankList));
    // Only support 8p comm group currently.
    if (!IsSingleNodeCommGroup(rank_list)) {
      continue;
    }
    const std::vector<TypeId> valid_types = {kFloat16->type_id(), kBFloat16->type_id()};
    if (!IsNodesDTypeSameAndValid({all_gather_cnode->input(kIndex1), matmul_cnode->input(kIndex2)}, valid_types)) {
      continue;
    }

    auto all_gather_matmul_prim = prim::kPrimAllGatherMatmul->Clone();
    all_gather_matmul_prim->AddAttr(kAttrGroup, all_gather_prim->GetAttr(kAttrGroup));
    all_gather_matmul_prim->AddAttr(kAttrRankSize, all_gather_prim->GetAttr(kAttrRankSize));
    all_gather_matmul_prim->AddAttr(kAttrCommTurn, MakeValue<int64_t>(0));     // default value: 0
    all_gather_matmul_prim->AddAttr(kAttrGatherIndex, MakeValue<int64_t>(0));  // only support 0 currently
    all_gather_matmul_prim->AddAttr(kAttrIsTransA, matmul_prim->GetAttr(kAttrTransposeX1));
    all_gather_matmul_prim->AddAttr(kAttrIsTransB, matmul_prim->GetAttr(kAttrTransposeX2));

    auto all_gather_matmul_input1 = all_gather_cnode->input(kIndex1);
    auto all_gather_matmul_input2 = matmul_cnode->input(kIndex2);
    auto all_gather_matmul_cnode =
      func_graph->NewCNode({NewValueNode(all_gather_matmul_prim), all_gather_matmul_input1, all_gather_matmul_input2});
    auto matmul_cnode_dtype = common::AnfAlgo::GetOutputInferDataType(matmul_cnode, kIndex0);
    auto matmul_cnode_shape = common::AnfAlgo::GetOutputInferShape(matmul_cnode, kIndex0);
    auto all_gather_cnode_dtype = common::AnfAlgo::GetOutputInferDataType(all_gather_cnode, kIndex0);
    auto all_gather_cnode_shape = common::AnfAlgo::GetOutputInferShape(all_gather_cnode, kIndex0);
    common::AnfAlgo::SetOutputTypeAndDetailShape({matmul_cnode_dtype, all_gather_cnode_dtype},
                                                 {std::make_shared<abstract::Shape>(matmul_cnode_shape),
                                                  std::make_shared<abstract::Shape>(all_gather_cnode_shape)},
                                                 all_gather_matmul_cnode.get());

    // Replace graph
    auto prev_cnode = all_gather_cnode->input(kIndex1);
    manager->SetEdge(all_gather_matmul_cnode, kIndex1, all_gather_cnode->input(kIndex1));
    manager->SetEdge(all_gather_matmul_cnode, kIndex2, all_gather_matmul_input2);
    auto matmul_cnode_users = manager->node_users()[matmul_cnode];
    auto tuple_get_item_cnode_0 = func_graph->NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), all_gather_matmul_cnode, NewValueNode(MakeValue<int64_t>(0))});
    tuple_get_item_cnode_0->set_abstract(matmul_cnode->abstract());
    for (const auto &matmul_cnode_user_pair : matmul_cnode_users) {
      manager->SetEdge(matmul_cnode_user_pair.first, matmul_cnode_user_pair.second, tuple_get_item_cnode_0);
    }

    auto all_gather_cnode_users = manager->node_users()[all_gather_cnode];
    if (all_gather_cnode_users.size() > 0) {
      auto tuple_get_item_cnode_1 = func_graph->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), all_gather_matmul_cnode, NewValueNode(MakeValue<int64_t>(1))});
      tuple_get_item_cnode_1->set_abstract(all_gather_cnode->abstract());
      for (const auto &all_gather_cnode_user_pair : all_gather_cnode_users) {
        if (all_gather_cnode_user_pair.first != matmul_cnode) {
          manager->SetEdge(all_gather_cnode_user_pair.first, all_gather_cnode_user_pair.second, tuple_get_item_cnode_1);
        }
      }
    }
    MS_LOG(INFO) << "Fusion " << all_gather_cnode->fullname_with_scope() << " and "
                 << matmul_cnode->fullname_with_scope() << " succeed,";
  }
  return;
}

}  // namespace
void FusionMC2Op(const FuncGraphPtr &func_graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto soc_version = ms_context->ascend_soc_version();
  auto enable_fusion = (common::GetEnv("MS_DEV_ENABLE_MC2_FUSION") == "1");
  if (soc_version != "ascend910b" || !enable_fusion) {
    MS_LOG(INFO) << "The pass of FusionMC2 is only take effect on ascend910b, current soc version is " << soc_version
                 << ". Please export MS_DEV_ENABLE_MC2_FUSION=1 and run on 910B hardware.";
    return;
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  FuncGraphPtr forward_graph = func_graph;
  FuncGraphPtr backward_graph = func_graph;
  for (const auto &each_graph : manager->func_graphs()) {
    if (IsCellReuseForwardGraph(each_graph)) {
      forward_graph = each_graph;
      backward_graph = GetCellReuseBackwardGraph(forward_graph);
      if (backward_graph == nullptr) {
        MS_LOG(WARNING) << "Failed to find backward cell reuse graph, skip pass 'fusion_mc2_op'.";
        return;
      }
      break;
    }
  }
  if (forward_graph == backward_graph) {
    MatmulReduceScatterFusion(func_graph);
    AllGatherMatmulFusion(func_graph);
  } else {
    auto mc2_fusion_level = 3;
    auto mc2_fusion_level_str = common::GetEnv("MS_DEV_MC2_FUSION_LEVEL");
    if (!mc2_fusion_level_str.empty()) {
      mc2_fusion_level = std::stoi(mc2_fusion_level_str);
    }
    MS_LOG(INFO) << "MC2 fusion level: " << mc2_fusion_level;
    if (mc2_fusion_level == 1 || mc2_fusion_level == 3) {
      MatmulReduceScatterFusion(forward_graph);
      AllGatherMatmulFusion(forward_graph);
    }
    if (mc2_fusion_level == 2 || mc2_fusion_level == 3) {
      MatmulReduceScatterFusion(backward_graph);
      AllGatherMatmulFusion(backward_graph);
    }
  }
  return;
}
}  // namespace parallel
}  // namespace mindspore
