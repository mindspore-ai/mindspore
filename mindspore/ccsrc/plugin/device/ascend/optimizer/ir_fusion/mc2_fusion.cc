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
#include "plugin/device/ascend/optimizer/ir_fusion/mc2_fusion.h"
#include <memory>
#include <algorithm>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "ops/math_ops.h"
#include "ops/other_ops.h"

namespace mindspore::opt {
namespace {
enum MC2FusionLevel {
  kMC2NotFusion = 0,
  kMC2FusionForward = 1,
  kMC2FusionBackward = 2,
  kMC2FusionFull,
};
bool IsForwardNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  return !(cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) || cnode->HasAttr(kAttrDuplicated));
}

bool IsRecomputeNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  return cnode->HasAttr(kAttrDuplicated);
}

bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find("Gradients") == 0;
}

// Return true if rank_ids is a continuous 8p group.
bool IsSingleNodeCommGroup(const std::vector<uint32_t> &rank_ids) {
  auto group_size = rank_ids.size();
  if (group_size != kSizeEight) {
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
}  // namespace
const BaseRef MC2FusionBase::DefinePattern() const {
  VectorRef pattern = DefineFusionPattern();
  return pattern;
}

const AnfNodePtr MC2FusionBase::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto mc2_fusion_level = ms_context->get_param<int>(MS_CTX_COMPUTE_COMMUNICATE_FUSION_LEVEL);
  if (mc2_fusion_level == kMC2NotFusion) {
    MS_LOG(DEBUG) << "MC2 fusion level is 0, not enable fusion.";
    return nullptr;
  }

  MS_LOG(DEBUG) << "Do " << name() << " fusion.";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(DEBUG) << "Func graph, node and equiv should be not nullptr, but some of them are nullptr";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(DEBUG) << "Node should be cnode, but it is not cnode.";
    return nullptr;
  }
  if (mc2_fusion_level == kMC2FusionForward && !IsForwardNode(node)) {
    MS_LOG(DEBUG) << "MC2 fusion level is " << kMC2FusionForward << ", only apply to forward node. Skip node "
                  << node->fullname_with_scope();
    return nullptr;
  }
  if (mc2_fusion_level == kMC2FusionBackward && !(IsBpropNode(node) || IsRecomputeNode(node))) {
    MS_LOG(DEBUG) << "MC2 fusion level is " << kMC2FusionBackward << ", only apply to backward node. Skip node "
                  << node->fullname_with_scope();
    return nullptr;
  }

  auto fusion_node = CreateFusionCNode(func_graph, node, equiv);
  if (fusion_node == nullptr) {
    MS_LOG(DEBUG) << node->fullname_with_scope() << " not fusion.";
    return nullptr;
  }
  MS_LOG(DEBUG) << node->fullname_with_scope() << " fusion success, node name: " << fusion_node->fullname_with_scope();
  return fusion_node;
}

const VectorRef MatmulReduceScatterFusion::DefineFusionPattern() const {
  MS_LOG(DEBUG) << "Do MatmulReduceScatterPattern.";
  // MatMul
  auto x_input = std::make_shared<Var>();  // input x
  auto w_input = std::make_shared<Var>();  // input w
  MS_CHECK_TRUE_RET(w_input != nullptr, {});
  MS_CHECK_TRUE_RET(x_input != nullptr, {});

  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});

  auto matmul_x_w = VectorRef({is_matmul, x_input, w_input});

  // ReduceScatter
  auto is_reduce_scatter = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceScatter>);
  MS_CHECK_TRUE_RET(is_reduce_scatter != nullptr, {});
  auto reduce_scatter = VectorRef({is_reduce_scatter, matmul_x_w});
  return reduce_scatter;
}

CNodePtr MatmulReduceScatterFusion::CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "Create MatmulReduceScatter CNode";
  auto reduce_scatter_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reduce_scatter_cnode != nullptr, {});

  auto matmul_cnode = reduce_scatter_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == reduce_scatter_cnode->func_graph(), {});

  auto input_x = matmul_cnode->input(kIndex1);
  auto input_w = matmul_cnode->input(kIndex2);
  std::vector<TypeId> valid_type_list = {kFloat16->type_id(), kBFloat16->type_id()};
  MS_CHECK_TRUE_RET(IsNodesDTypeSameAndValid({input_x, input_w}, valid_type_list), {});

  auto matmul_cnode_users = matmul_cnode->func_graph()->manager()->node_users()[matmul_cnode];
  MS_CHECK_TRUE_RET(matmul_cnode_users.size() == 1, {});

  // create op
  auto matmul_reduce_scatter_prim = prim::kPrimMatmulReduceScatter->Clone();
  MS_CHECK_TRUE_RET(matmul_reduce_scatter_prim, {});

  // add attr
  auto reduce_scatter_prim = GetCNodePrimitive(reduce_scatter_cnode);
  auto rank_list_attr = reduce_scatter_prim->GetAttr(kAttrRankList);
  MS_CHECK_TRUE_RET(rank_list_attr != nullptr, {});
  auto rank_list = GetValue<std::vector<uint32_t>>(rank_list_attr);
  // Only support 8p comm group currently.
  MS_CHECK_TRUE_RET(IsSingleNodeCommGroup(rank_list), {});

  auto matmul_prim = GetCNodePrimitive(matmul_cnode);
  matmul_reduce_scatter_prim->AddAttr(kAttrGroup, reduce_scatter_prim->GetAttr(kAttrGroup));
  matmul_reduce_scatter_prim->AddAttr(kAttrRankSize, reduce_scatter_prim->GetAttr(kAttrRankSize));
  matmul_reduce_scatter_prim->AddAttr(kAttrReduceOp, reduce_scatter_prim->GetAttr(kAttrOp));
  matmul_reduce_scatter_prim->AddAttr(kAttrRankList, rank_list_attr);
  matmul_reduce_scatter_prim->AddAttr(kAttrCommTurn, MakeValue<int64_t>(0));  // default value: 0
  matmul_reduce_scatter_prim->AddAttr(kAttrIsTransA, matmul_prim->GetAttr(kAttrTransposeX1));
  matmul_reduce_scatter_prim->AddAttr(kAttrIsTransB, matmul_prim->GetAttr(kAttrTransposeX2));

  auto matmul_reduce_scatter_cnode = func_graph->NewCNode({NewValueNode(matmul_reduce_scatter_prim), input_x, input_w});
  if (matmul_reduce_scatter_cnode == nullptr) {
    MS_LOG(DEBUG) << "New matmul_reduce_scatter_cnode should not be null, but it is null.";
    return nullptr;
  }
  matmul_reduce_scatter_cnode->set_abstract(reduce_scatter_cnode->abstract()->Clone());
  matmul_reduce_scatter_cnode->set_fullname_with_scope(reduce_scatter_cnode->fullname_with_scope() +
                                                       "_matmul_reduce_scatter");
  MS_LOG(DEBUG) << "Create MatmulReduceScatter cnode success.";
  return matmul_reduce_scatter_cnode;
}

const VectorRef AllGatherMatmulFusion::DefineFusionPattern() const {
  MS_LOG(DEBUG) << "Do AllGatherMatmulPattern.";
  // MatMul
  auto x_input = std::make_shared<Var>();  // input x
  auto w_input = std::make_shared<Var>();  // input w
  MS_CHECK_TRUE_RET(w_input != nullptr, {});
  MS_CHECK_TRUE_RET(x_input != nullptr, {});

  auto is_allgather = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllGather>);
  MS_CHECK_TRUE_RET(is_allgather != nullptr, {});

  auto allgather_x = VectorRef({is_allgather, x_input});

  // ReduceScatter
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto matmul = VectorRef({is_matmul, allgather_x, w_input});
  return matmul;
}

CNodePtr AllGatherMatmulFusion::CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "Create AllGatherMatmul CNode";
  auto matmul_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});

  auto all_gather_cnode = matmul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(all_gather_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(all_gather_cnode->func_graph() == matmul_cnode->func_graph(), {});

  auto input_x = all_gather_cnode->input(kIndex1);
  auto input_w = matmul_cnode->input(kIndex2);
  std::vector<TypeId> valid_type_list = {kFloat16->type_id(), kBFloat16->type_id()};
  MS_CHECK_TRUE_RET(IsNodesDTypeSameAndValid({input_x, input_w}, valid_type_list), {});

  // create op
  auto all_gather_matmul_prim = prim::kPrimAllGatherMatmul->Clone();
  MS_CHECK_TRUE_RET(all_gather_matmul_prim, {});

  // add attr
  auto all_gather_prim = GetCNodePrimitive(all_gather_cnode);
  auto rank_list_attr = all_gather_prim->GetAttr(kAttrRankList);
  MS_CHECK_TRUE_RET(rank_list_attr != nullptr, {});
  auto rank_list = GetValue<std::vector<uint32_t>>(rank_list_attr);
  // Only support 8p comm group currently.
  MS_CHECK_TRUE_RET(IsSingleNodeCommGroup(rank_list), {});

  auto matmul_prim = GetCNodePrimitive(matmul_cnode);
  all_gather_matmul_prim->AddAttr(kAttrGroup, all_gather_prim->GetAttr(kAttrGroup));
  all_gather_matmul_prim->AddAttr(kAttrRankSize, all_gather_prim->GetAttr(kAttrRankSize));
  all_gather_matmul_prim->AddAttr(kAttrRankList, rank_list_attr);
  all_gather_matmul_prim->AddAttr(kAttrCommTurn, MakeValue<int64_t>(0));     // default value: 0
  all_gather_matmul_prim->AddAttr(kAttrGatherIndex, MakeValue<int64_t>(0));  // only support 0 currently
  all_gather_matmul_prim->AddAttr(kAttrIsTransA, matmul_prim->GetAttr(kAttrTransposeX1));
  all_gather_matmul_prim->AddAttr(kAttrIsTransB, matmul_prim->GetAttr(kAttrTransposeX2));

  auto all_gather_matmul_cnode = func_graph->NewCNode({NewValueNode(all_gather_matmul_prim), input_x, input_w});
  if (all_gather_matmul_cnode == nullptr) {
    MS_LOG(DEBUG) << "New all_gather_matmul_cnode should not be null, but it is null.";
    return nullptr;
  }

  // Set abstract
  auto matmul_cnode_dtype = common::AnfAlgo::GetOutputInferDataType(matmul_cnode, kIndex0);
  auto matmul_cnode_shape = common::AnfAlgo::GetOutputInferShape(matmul_cnode, kIndex0);
  auto all_gather_cnode_dtype = common::AnfAlgo::GetOutputInferDataType(all_gather_cnode, kIndex0);
  auto all_gather_cnode_shape = common::AnfAlgo::GetOutputInferShape(all_gather_cnode, kIndex0);
  common::AnfAlgo::SetOutputTypeAndDetailShape(
    {matmul_cnode_dtype, all_gather_cnode_dtype},
    {std::make_shared<abstract::Shape>(matmul_cnode_shape), std::make_shared<abstract::Shape>(all_gather_cnode_shape)},
    all_gather_matmul_cnode.get());
  all_gather_matmul_cnode->set_fullname_with_scope(matmul_cnode->fullname_with_scope() + "_all_gather_matmul_scatter");

  // Extract process for node
  auto manager = all_gather_cnode->func_graph()->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, {});

  // Insert TupleGetItem After MatMul
  auto matmul_cnode_users = manager->node_users()[matmul_cnode];
  auto tuple_get_item_cnode_0 =
    func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), matmul_cnode, NewValueNode(MakeValue<int64_t>(0))});
  tuple_get_item_cnode_0->set_abstract(matmul_cnode->abstract());
  for (const auto &matmul_cnode_user_pair : matmul_cnode_users) {
    manager->SetEdge(matmul_cnode_user_pair.first, matmul_cnode_user_pair.second, tuple_get_item_cnode_0);
  }

  // Replace other node
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

  MS_LOG(DEBUG) << "Create AllGatherMatmul cnode success.";
  return all_gather_matmul_cnode;
}
}  // namespace mindspore::opt
