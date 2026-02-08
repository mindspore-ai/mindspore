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
#include "backend/common/pass/ir_fusion/mc2_fusion.h"
#include <set>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include "base/base.h"
#include "backend/common/pass/common/gllo_utils.h"
#include "include/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "primitive/math_ops.h"
#include "primitive/other_ops.h"
#include "primitive/lite_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_q.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::opt {
namespace {

bool IsForwardNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return !(cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) || cnode->HasAttr(kAttrDuplicated));
}

bool IsRecomputeNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return cnode->HasAttr(kAttrDuplicated);
}

bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find("Gradients") == 0;
}

bool IsOnlyOnePathToMatmul(const CNodePtr &all_gather_cnode, const CNodePtr &matmul_cnode,
                           const FuncGraphPtr &func_graph) {
  std::unordered_set<CNodePtr> visited;
  int path_count = 0;

  std::function<void(const CNodePtr &)> dfs = [&](const CNodePtr &current_node) {
    if (path_count > 1) return;

    if (visited.count(current_node) > 0) {
      return;
    }
    visited.insert(current_node);

    if (current_node == matmul_cnode) {
      path_count++;
      return;
    }

    auto manager = current_node->func_graph()->manager();
    if (manager != nullptr) {
      auto node_users = manager->node_users()[current_node];
      for (const auto &user_pair : node_users) {
        dfs(user_pair.first->cast<CNodePtr>());
      }
    }
  };

  auto manager = all_gather_cnode->func_graph()->manager();
  auto all_gather_users = manager->node_users()[all_gather_cnode];

  for (const auto &user_pair : all_gather_users) {
    visited.clear();
    dfs(user_pair.first->cast<CNodePtr>());
  }

  return path_count == 1;
}

bool IsKbkMode(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return !kernel_graph->is_graph_run_mode();
}

ShapeVector GetShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto base_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr->shape();
  }
  return {};
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

template <typename T>
T GetInputValueFromCNode(const CNodePtr &cnode, size_t index) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto inputs = cnode->inputs();
  if (index >= inputs.size()) {
    MS_LOG(EXCEPTION) << "The input index (" << index << ") is exceed of inputs size (" << inputs.size() << ").";
  }
  auto input_node = inputs[index];
  MS_EXCEPTION_IF_NULL(input_node);
  if (!input_node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "The " << index << "-th input is not a value node.";
  }
  auto value_ptr = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto value = value_ptr->value();
  MS_EXCEPTION_IF_NULL(value);
  return GetValue<T>(value);
}
}  // namespace
const BaseRef MC2FusionBase::DefinePattern() const {
  VectorRef pattern = DefineFusionPattern();
  return pattern;
}

const AnfNodePtr MC2FusionBase::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "Do " << name() << " fusion.";
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    MS_LOG(DEBUG) << "Func graph, node and equiv should be not nullptr, but some of them are nullptr";
    return nullptr;
  }
  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(DEBUG) << "Node should be cnode, but it is not cnode.";
    return nullptr;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto mc2_fusion_level = ms_context->get_param<int>(MS_CTX_COMPUTE_COMMUNICATE_FUSION_LEVEL);
  if (mc2_fusion_level == kMC2NotFusion) {
    MS_LOG(DEBUG) << "MC2 fusion level is 0, not enable fusion.";
    return nullptr;
  }

  if (common::IsExecuteSimulation()) {
    MS_LOG(EXCEPTION) << "Not support compute_communication_fusion_level when MS_SIMULATION_LEVEL=3.";
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

  if (IsKbkMode(func_graph)) {
    if (mc2_fusion_level != kMC2NotFusion && mc2_fusion_level != kMC2FusionForward &&
        mc2_fusion_level != kMC2FusionBackward && mc2_fusion_level != kMC2FusionFull) {
      MS_LOG(DEBUG) << "In KBK mode MC2 fusion level is " << mc2_fusion_level << ", only support 0, 1, 2, 3.";
      return nullptr;
    }
  }

  auto fusion_node = CreateFusionCNode(func_graph, node, equiv);
  if (fusion_node == nullptr) {
    MS_LOG(DEBUG) << node->fullname_with_scope() << " not fusion.";
    return nullptr;
  }
  MS_LOG(DEBUG) << node->fullname_with_scope() << " fusion success, node name: " << fusion_node->fullname_with_scope();
  return fusion_node;
}

std::vector<std::string> MatmulReduceScatterFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimMatMul->name(), prim::kPrimReduceScatter->name()};
  return ret;
}

const VectorRef MatmulReduceScatterFusion::DefineFusionPattern() const {
  MS_LOG(DEBUG) << "Do MatmulReduceScatterPattern.";
  // MatMul
  auto x_input = std::make_shared<Var>();       // input x
  auto w_input = std::make_shared<Var>();       // input w
  auto transpose_x1 = std::make_shared<Var>();  // transpose_x1
  auto transpose_x2 = std::make_shared<Var>();  // transpose_x2
  MS_CHECK_TRUE_RET(w_input != nullptr, {});
  MS_CHECK_TRUE_RET(x_input != nullptr, {});
  MS_CHECK_TRUE_RET(transpose_x1 != nullptr, {});
  MS_CHECK_TRUE_RET(transpose_x2 != nullptr, {});

  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});

  auto matmul_x_w = VectorRef({is_matmul, x_input, w_input, transpose_x1, transpose_x2});

  // ReduceScatter
  auto is_reduce_scatter = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReduceScatter>);
  MS_CHECK_TRUE_RET(is_reduce_scatter != nullptr, {});
  auto reduce_scatter = VectorRef({is_reduce_scatter, matmul_x_w});
  return reduce_scatter;
}

CNodePtr MatmulReduceScatterFusion::CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "Create MatmulReduceScatter CNode";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  const auto &soc_version = context_ptr->ascend_soc_version();
  if (soc_version == "ascend910_93") {
    // The current feature is not supported on this machine.
    return nullptr;
  }

  auto reduce_scatter_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(reduce_scatter_cnode != nullptr, {});

  auto matmul_cnode = reduce_scatter_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == reduce_scatter_cnode->func_graph(), {});

  auto input = matmul_cnode->input(kIndex1);
  auto x2 = matmul_cnode->input(kIndex2);
  std::vector<TypeId> valid_type_list = {kFloat16->type_id(), kBFloat16->type_id()};
  MS_CHECK_TRUE_RET(IsNodesDTypeSameAndValid({input, x2}, valid_type_list), {});

  auto matmul_cnode_users = matmul_cnode->func_graph()->manager()->node_users()[matmul_cnode];

  for (const auto &user : matmul_cnode_users) {
    if (user.first != reduce_scatter_cnode && !IsPrimitiveCNode(user.first, prim::kPrimUpdateState)) {
      return {};
    }
  }

  auto trans_input = GetInputValueFromCNode<bool>(matmul_cnode, kIndex3);
  auto trans_x2 = GetInputValueFromCNode<bool>(matmul_cnode, kIndex4);
  // Ensure trans_input is false for mc2 and lccl fusion
  MS_CHECK_TRUE_RET(!trans_input, {});

  auto input_shape = GetShape(input);
  MS_CHECK_TRUE_RET(input_shape.size() == kSizeTwo, {});
  MS_CHECK_TRUE_RET(GetShape(x2).size() == kSizeTwo, {});

  constexpr int64_t kMaxValue = 65535;
  constexpr int64_t kMinValue = 256;
  int64_t input_dim1 = input_shape[kIndex1];
  if (input_dim1 >= kMaxValue || input_dim1 < kMinValue) {
    MS_LOG(WARNING) << "The second dimension of input is " << input_dim1
                    << ", but aclnnMatmulReduceScatter required should be between " << kMinValue << " (inclusive) and "
                    << kMaxValue << " (exclusive).";
    return nullptr;
  }

  auto matmul_reduce_scatter_prim = prim::kPrimMatmulReduceScatter->Clone();
  MS_CHECK_TRUE_RET(matmul_reduce_scatter_prim, {});
  auto reduce_scatter_prim = GetCNodePrimitive(reduce_scatter_cnode);
  MS_EXCEPTION_IF_NULL(reduce_scatter_prim);
  matmul_reduce_scatter_prim->AddAttr(kAttrGroup, reduce_scatter_prim->GetAttr(kAttrGroup));
  auto reduce_op = GetValue<std::string>(reduce_scatter_prim->GetAttr(kAttrOp));
  if (reduce_op != "sum") {
    MS_LOG(WARNING) << "The reduce op is " << reduce_op << ", but aclnnMatmulReduceScatter only support sum.";
    return nullptr;
  }
  matmul_reduce_scatter_prim->AddAttr(kAttrOp, reduce_scatter_prim->GetAttr(kAttrOp));

  auto kernel_graph = func_graph->cast<std::shared_ptr<mindspore::session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto matmul_reduce_scatter_cnode = NewCNode(
    {NewValueNode(matmul_reduce_scatter_prim), input, x2,
     kernel_graph->NewValueNode(reduce_scatter_prim->GetAttr(kAttrGroup)),
     kernel_graph->NewValueNode(reduce_scatter_prim->GetAttr(kAttrRankSize)),
     kernel_graph->NewValueNode(MakeValue<int64_t>(Reduction::REDUCTION_SUM)),
     kernel_graph->NewValueNode(mindspore::kNone), kernel_graph->NewValueNode(MakeValue<int64_t>(0)),
     kernel_graph->NewValueNode(MakeValue<bool>(trans_input)), kernel_graph->NewValueNode(MakeValue<bool>(trans_x2))},
    func_graph);
  if (matmul_reduce_scatter_cnode == nullptr) {
    MS_LOG(DEBUG) << "New matmul_reduce_scatter_cnode should not be null, but it is null.";
    return nullptr;
  }
  matmul_reduce_scatter_cnode->set_abstract(reduce_scatter_cnode->abstract()->Clone());
  matmul_reduce_scatter_cnode->set_scope(reduce_scatter_cnode->scope());
  for (const auto &matmul_cnode_user_pair : matmul_cnode_users) {
    if (common::AnfAlgo::CheckPrimitiveType(matmul_cnode_user_pair.first, prim::kPrimUpdateState)) {
      matmul_cnode->func_graph()->manager()->SetEdge(matmul_cnode_user_pair.first, matmul_cnode_user_pair.second,
                                                     matmul_reduce_scatter_cnode);
    }
  }
  MS_LOG(DEBUG) << "Create MatmulReduceScatter cnode success.";
  return matmul_reduce_scatter_cnode;
}

std::vector<std::string> AllGatherMatmulFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimAllGather->name(), prim::kPrimMatMul->name()};
  return ret;
}

const VectorRef AllGatherMatmulFusion::DefineFusionPattern() const {
  MS_LOG(DEBUG) << "Do AllGatherMatmulPattern.";
  // MatMul
  auto x_input = std::make_shared<Var>();       // input x
  auto w_input = std::make_shared<Var>();       // input w
  auto transpose_x1 = std::make_shared<Var>();  // transpose_x1
  auto transpose_x2 = std::make_shared<Var>();  // transpose_x2
  MS_CHECK_TRUE_RET(w_input != nullptr, {});
  MS_CHECK_TRUE_RET(x_input != nullptr, {});
  MS_CHECK_TRUE_RET(transpose_x1 != nullptr, {});
  MS_CHECK_TRUE_RET(transpose_x2 != nullptr, {});

  auto is_allgather = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllGather>);
  MS_CHECK_TRUE_RET(is_allgather != nullptr, {});

  auto allgather_x = VectorRef({is_allgather, x_input});

  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto matmul = VectorRef({is_matmul, allgather_x, w_input, transpose_x1, transpose_x2});
  return matmul;
}

CNodePtr AllGatherMatmulFusion::CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "Create AllGatherMatmul CNode";
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  const auto &soc_version = context_ptr->ascend_soc_version();
  if (soc_version == "ascend310p" || soc_version == "ascend910_93") {
    // The current feature is not supported on this machine.
    return nullptr;
  }
  auto matmul_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});

  auto all_gather_cnode = matmul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(all_gather_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(all_gather_cnode->func_graph() == matmul_cnode->func_graph(), {});

  if (!IsOnlyOnePathToMatmul(all_gather_cnode, matmul_cnode, func_graph)) {
    return nullptr;
  }

  auto input = all_gather_cnode->input(kIndex1);
  auto x2 = matmul_cnode->input(kIndex2);
  std::vector<TypeId> valid_type_list = {kFloat16->type_id(), kBFloat16->type_id()};
  MS_CHECK_TRUE_RET(IsNodesDTypeSameAndValid({input, x2}, valid_type_list), {});

  auto input_shape = GetShape(input);
  MS_CHECK_TRUE_RET(input_shape.size() == kSizeTwo, {});
  MS_CHECK_TRUE_RET(GetShape(x2).size() == kSizeTwo, {});

  constexpr int64_t kMaxValue = 65535;
  constexpr int64_t kMinValue = 256;
  int64_t input_dim1 = input_shape[kIndex1];
  if (input_dim1 >= kMaxValue || input_dim1 < kMinValue) {
    MS_LOG(WARNING) << "The second dimension of input is " << input_dim1
                    << ", but aclnnAllGatherMatmul required should be between " << kMinValue << " (inclusive) and "
                    << kMaxValue << " (exclusive).";
    return nullptr;
  }

  auto trans_input = GetInputValueFromCNode<bool>(matmul_cnode, kIndex3);
  auto trans_x2 = GetInputValueFromCNode<bool>(matmul_cnode, kIndex4);
  // Ensure trans_input is false for mc2 and lccl fusion
  MS_CHECK_TRUE_RET(!trans_input, {});

  // create op
  auto all_gather_matmul_prim = prim::kPrimAllGatherMatmul->Clone();
  MS_CHECK_TRUE_RET(all_gather_matmul_prim, {});
  auto all_gather_prim = GetCNodePrimitive(all_gather_cnode);
  MS_EXCEPTION_IF_NULL(all_gather_prim);
  all_gather_matmul_prim->AddAttr(kAttrGroup, all_gather_prim->GetAttr(kAttrGroup));
  auto kernel_graph = func_graph->cast<std::shared_ptr<mindspore::session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto all_gather_matmul_cnode = NewCNode(
    {NewValueNode(all_gather_matmul_prim), input, x2, kernel_graph->NewValueNode(all_gather_prim->GetAttr(kAttrGroup)),
     kernel_graph->NewValueNode(all_gather_prim->GetAttr(kAttrRankSize)), kernel_graph->NewValueNode(mindspore::kNone),
     kernel_graph->NewValueNode(MakeValue<int64_t>(0)), kernel_graph->NewValueNode(MakeValue<bool>(true)),
     kernel_graph->NewValueNode(MakeValue<int64_t>(0)), kernel_graph->NewValueNode(MakeValue<bool>(trans_input)),
     kernel_graph->NewValueNode(MakeValue<bool>(trans_x2))},
    func_graph);
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
  all_gather_matmul_cnode->set_scope(matmul_cnode->scope());

  // Extract process for node
  auto manager = all_gather_cnode->func_graph()->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, {});

  // Insert TupleGetItem After MatMul
  auto matmul_cnode_users = manager->node_users()[matmul_cnode];
  auto tuple_get_item_cnode_0 =
    NewCNode({NewValueNode(prim::kPrimTupleGetItem), matmul_cnode, NewValueNode(MakeValue<int64_t>(0))}, func_graph);
  tuple_get_item_cnode_0->set_abstract(matmul_cnode->abstract());
  for (const auto &matmul_cnode_user_pair : matmul_cnode_users) {
    manager->SetEdge(matmul_cnode_user_pair.first, matmul_cnode_user_pair.second, tuple_get_item_cnode_0);
  }

  // Replace other node
  auto all_gather_cnode_users = manager->node_users()[all_gather_cnode];
  if (all_gather_cnode_users.size() > kSizeOne) {
    auto tuple_get_item_cnode_1 =
      NewCNode({NewValueNode(prim::kPrimTupleGetItem), all_gather_matmul_cnode, NewValueNode(MakeValue<int64_t>(1))},
               func_graph);
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

std::vector<std::string> QuantBatchMatmulAllReduceFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimQuantBatchMatmul->name(), prim::kPrimAllReduce->name()};
  return ret;
}

const VectorRef QuantBatchMatmulAllReduceFusion::DefineFusionPattern() const {
  MS_LOG(DEBUG) << "Do QuantBatchMatmulAllReducePattern.";
  // QuantBatchMatMul
  auto x = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(x != nullptr, {});
  auto w = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(w != nullptr, {});
  auto scale = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(scale != nullptr, {});
  auto offset = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(offset != nullptr, {});
  auto bias = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(bias != nullptr, {});
  auto pertoken_scale = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(pertoken_scale != nullptr, {});
  auto trans_a = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_a != nullptr, {});
  auto trans_b = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(trans_b != nullptr, {});
  auto out_dtype = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(out_dtype != nullptr, {});

  auto is_quant_batch_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantBatchMatmul>);
  MS_CHECK_TRUE_RET(is_quant_batch_matmul != nullptr, {});

  auto quant_batch_matmul =
    VectorRef({is_quant_batch_matmul, x, w, scale, offset, bias, pertoken_scale, trans_a, trans_b, out_dtype});

  // AllReduce
  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  auto allreduce = VectorRef({is_allreduce, quant_batch_matmul});
  return allreduce;
}

CNodePtr QuantBatchMatmulAllReduceFusion::CreateFusionCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                            const EquivPtr &equiv) const {
  if (!IsKbkMode(func_graph)) {
    MS_LOG(DEBUG) << "QuantBatchMatmulAllReduceFusion only support in KBK mode.";
    return nullptr;
  }
  MS_LOG(DEBUG) << "Create QuantBatchMatmulAllReduce CNode";
  auto allreduce_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(allreduce_cnode != nullptr, {});

  auto qbmm_cnode = allreduce_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(qbmm_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(qbmm_cnode->func_graph() == allreduce_cnode->func_graph(), {});

  auto pertoken_scale = qbmm_cnode->input(kIndex6);
  if (IsValueNode<None>(pertoken_scale)) {
    MS_LOG(INFO) << "The pass only to fuse qbmm(pertoken) with communication.";
    return nullptr;
  }
  auto offset = qbmm_cnode->input(kIndex4);
  auto bias = qbmm_cnode->input(kIndex5);
  if (!IsValueNode<None>(offset) || !IsValueNode<None>(bias)) {
    MS_LOG(INFO) << "The pass only support qbmm's offset and bias must be None.";
    return nullptr;
  }

  auto qbmm_cnode_users = qbmm_cnode->func_graph()->manager()->node_users()[qbmm_cnode];
  MS_CHECK_TRUE_RET(qbmm_cnode_users.size() == 1, {});

  // create op
  auto qbmm_reduce_prim = prim::kPrimQuantBatchMatmulAllReduce->Clone();
  MS_CHECK_TRUE_RET(qbmm_reduce_prim, {});

  auto transpose_a_node = qbmm_cnode->input(kIndex7)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(transpose_a_node != nullptr, {});
  auto transpose_b_node = qbmm_cnode->input(kIndex8)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(transpose_b_node != nullptr, {});
  // add attr
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  MS_CHECK_TRUE_RET(allreduce_prim != nullptr, {});
  qbmm_reduce_prim->AddAttr(kAttrGroup, allreduce_prim->GetAttr(kAttrGroup));
  qbmm_reduce_prim->AddAttr(kAttrFusion, allreduce_prim->GetAttr(kAttrFusion));
  qbmm_reduce_prim->AddAttr(kAttrOp, allreduce_prim->GetAttr(kAttrOp));
  qbmm_reduce_prim->AddAttr(kAttrIsTransA, transpose_a_node->value());
  qbmm_reduce_prim->AddAttr(kAttrIsTransB, transpose_b_node->value());

  auto input_x = qbmm_cnode->input(kIndex1);
  auto input_w = qbmm_cnode->input(kIndex2);
  auto x_shape = GetShape(input_x);
  auto w_shape = GetShape(input_w);
  MS_CHECK_TRUE_RET(x_shape.size() == kSizeTwo && w_shape.size() == kSizeTwo, {});
  auto scale = qbmm_cnode->input(kIndex3);
  auto out_type = qbmm_cnode->input(kIndex9);
  auto qbmm_allreduce_cnode = NewCNode(
    {NewValueNode(qbmm_reduce_prim), input_x, input_w, bias, offset, scale, pertoken_scale, out_type}, func_graph);
  if (qbmm_allreduce_cnode == nullptr) {
    MS_LOG(DEBUG) << "New qbmm_allreduce_cnode should not be null, but it is null.";
    return nullptr;
  }
  qbmm_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  qbmm_allreduce_cnode->set_scope(allreduce_cnode->scope());
  MS_LOG(DEBUG) << "Create QuantBatchMatmulAllReduce cnode success.";
  return qbmm_allreduce_cnode;
}
}  // namespace mindspore::opt
