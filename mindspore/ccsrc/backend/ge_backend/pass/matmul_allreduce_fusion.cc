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

#include "backend/ge_backend/pass/matmul_allreduce_fusion.h"
#include <set>
#include <vector>
#include <string>
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "primitive/other_ops.h"
#include "primitive/lite_ops.h"
#include "utils/ms_context.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "mindspore/ops/infer/ops_func_impl/communication/all_reduce.h"
#include "backend/common/pass/common/gllo_utils.h"
#include "ops_utils/op_utils.h"
#include "primitive/other_op_name.h"
#include "ir/anf.h"
#include "utils/phase.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::opt {
enum MC2FusionLevel { kMC2NotFusion = 0, kMC2FusionForward = 1, kMC2FusionBackward = 2, kMC2FusionFull = 3 };
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
  auto value = input_node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(value);
  return GetValue<T>(value);
}

ShapeVector GetShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->abstract()->GetShape();
  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr->shape();
  }
  return {};
}

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

bool IsKbkAclnnMode(const KernelGraphPtr &kernel_graph) {
  bool is_k_by_k_mode = !kernel_graph->is_graph_run_mode();
  bool enable_lccl = device::ascend::AscendHalManager::GetInstance().EnableLccl();
  //  When lccl communication is not enabled in the kbk scenario
  return is_k_by_k_mode && !enable_lccl;
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

std::vector<std::string> MatMulAllReduceFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimMatMul->name(), prim::kPrimAllReduce->name()};
  return ret;
}

const BaseRef MatMulAllReduceFusion::DefinePattern() const {
  auto matmul_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_1 != nullptr, {});
  auto matmul_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_2 != nullptr, {});
  auto transpose_a = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(transpose_a != nullptr, {});
  auto transpose_b = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(transpose_b != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  VectorRef matmul_ref = VectorRef({is_matmul, matmul_input_1, matmul_input_2, transpose_a, transpose_b});

  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_allreduce, matmul_ref});
  return pattern_ref;
}

PrimitivePtr MatMulAllReduceFusion::CreateMatMulAllReducePrim(const PrimitivePtr &allreduce_prim,
                                                              const CNodePtr &matmul_node) const {
  // create op
  auto matmul_allreduce_prim = prim::kPrimMatMulAllReduce->Clone();
  MS_CHECK_TRUE_RET(matmul_allreduce_prim, {});
  auto transpose_a_node = matmul_node->input(kIndex3)->cast<ValueNodePtr>();
  auto transpose_b_node = matmul_node->input(kIndex4)->cast<ValueNodePtr>();
  // add attr
  matmul_allreduce_prim->AddAttr(kAttrNameGroup, allreduce_prim->GetAttr(kAttrNameGroup));
  matmul_allreduce_prim->AddAttr(kAttrNameFusion, allreduce_prim->GetAttr(kAttrNameFusion));
  matmul_allreduce_prim->AddAttr(kAttrNameOp, allreduce_prim->GetAttr(kAttrNameOp));
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeA, transpose_a_node->value());
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeB, transpose_b_node->value());
  return matmul_allreduce_prim;
}

AnfNodePtr MatMulAllReduceFusion::CreateMatMulAllReduceNode(const FuncGraphPtr &func_graph,
                                                            const AnfNodePtr &node) const {
  MS_LOG(DEBUG) << "start create MatMulAllReduce";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto allreduce_cnode = node->cast<CNodePtr>();
  MS_ASSERT(allreduce_cnode != nullptr);
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  MS_EXCEPTION_IF_NULL(allreduce_prim);
  auto matmul_cnode = allreduce_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_ASSERT(matmul_cnode != nullptr);
  auto input_x_node = matmul_cnode->input(kIndex1);
  MS_ASSERT(input_x_node != nullptr);
  auto input_y_node = matmul_cnode->input(kIndex2);
  MS_ASSERT(input_y_node != nullptr);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  if (IsKbkAclnnMode(kernel_graph)) {
    auto is_trans_a = GetInputValueFromCNode<bool>(matmul_cnode, kIndex3);
    // current only support b tans
    MS_CHECK_TRUE_RET(!is_trans_a, {});

    // X1 supports two or three dimensions, and X2 supports only two dimensions
    MS_CHECK_TRUE_RET(GetShape(input_x_node).size() == kSizeTwo || GetShape(input_x_node).size() == kSizeThree, {});
    MS_CHECK_TRUE_RET(GetShape(input_y_node).size() == kSizeTwo, {});

    auto reduce_op = allreduce_prim->GetAttr(kAttrNameOp);
    // current only support "sum"
    MS_CHECK_TRUE_RET(reduce_op->isa<StringImm>() && reduce_op->cast<StringImmPtr>()->value() == "sum", {});

    //  Currently, the aclnn scenario under kbk only supports f16 and bf16
    std::vector<TypeId> valid_type_list = {kFloat16->type_id(), kBFloat16->type_id()};
    MS_CHECK_TRUE_RET(IsNodesDTypeSameAndValid({input_x_node, input_y_node}, valid_type_list), {});
  } else {
    const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
    if (!CheckSupportDataType(input_x_node, support_dtype)) {
      return nullptr;
    }
  }

  auto matmul_allreduce_prim_c = CreateMatMulAllReducePrim(allreduce_prim, matmul_cnode);
  std::vector<AnfNodePtr> matmul_allreduce_inputs = {input_x_node, input_y_node};

  auto matmul_allreduce_cnode = func_graph->NewCNode(matmul_allreduce_prim_c, matmul_allreduce_inputs);
  matmul_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  MS_LOG(DEBUG) << "create MatMulAllReduce success.";
  return matmul_allreduce_cnode;
}

bool IsSupportFusion(const mindspore::AnfNodePtr &node) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto mc2_fusion_level = ms_context->get_param<int>(MS_CTX_COMPUTE_COMMUNICATE_FUSION_LEVEL);
  if (mc2_fusion_level != kMC2NotFusion && mc2_fusion_level != kMC2FusionForward &&
      mc2_fusion_level != kMC2FusionBackward && mc2_fusion_level != kMC2FusionFull) {
    MS_LOG(DEBUG) << "In KBK mode MC2 fusion level is " << mc2_fusion_level << ", only support 0, 1, 2, 3.";
    return false;
  }
  if (mc2_fusion_level == kMC2NotFusion) {
    MS_LOG(DEBUG) << "MC2 fusion level is 0, not enable fusion.";
    return false;
  }

  if (mc2_fusion_level == kMC2FusionForward && !IsForwardNode(node)) {
    MS_LOG(DEBUG) << "MC2 fusion level is " << kMC2FusionForward << ", only apply to forward node. Skip node "
                  << node->fullname_with_scope();
    return false;
  }
  if (mc2_fusion_level == kMC2FusionBackward && !(IsBpropNode(node) || IsRecomputeNode(node))) {
    MS_LOG(DEBUG) << "MC2 fusion level is " << kMC2FusionBackward << ", only apply to backward node. Skip node "
                  << node->fullname_with_scope();
    return false;
  }
  if (common::IsExecuteSimulation()) {
    MS_LOG(EXCEPTION) << "Not support compute_communication_fusion_level when MS_SIMULATION_LEVEL=3.";
  }
  return true;
}

const AnfNodePtr MatMulAllReduceFusion::Process(const mindspore::FuncGraphPtr &func_graph,
                                                const mindspore::AnfNodePtr &node,
                                                const mindspore::EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (IsKbkAclnnMode(kernel_graph)) {
    if (!IsSupportFusion(node)) {
      return nullptr;
    }
  } else {
#ifndef ENABLE_INTERNAL_KERNELS
    return nullptr;
#else
    if (common::IsExecuteSimulation()) {
      MS_LOG(INFO) << "Not support MatMulAllReduceFusion when MS_SIMULATION_LEVEL=3.";
      return nullptr;
    }
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (!ms_context->IsEnableInferBoost()) {
      return nullptr;
    }

    auto phase = PhaseManager::GetInstance().phase();
    bool enable_lccl = device::ascend::AscendHalManager::GetInstance().EnableLccl();
    if (!enable_lccl || phase.rfind(kPhaseNamePrefill) == std::string::npos) {
      return nullptr;
    }

    auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
    bool enable_matmul_allreduce =
      (std::find(enable_op_list.begin(), enable_op_list.end(), kMatMulAllReduceOpName) != enable_op_list.end());
    if (!enable_matmul_allreduce) {
      return nullptr;
    }
#endif  // ENABLE_INTERNAL_KERNELS
  }

  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto allreduce_cnode = node->cast<CNodePtr>();
  if (allreduce_cnode->size() != kSizeTwo) {
    return nullptr;
  }

  auto matmul_allreduce_cnode = CreateMatMulAllReduceNode(func_graph, node);
  MS_CHECK_TRUE_RET(matmul_allreduce_cnode != nullptr, nullptr);

  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);

  // replace allreduce to MatMulAllReduce
  (void)manager->Replace(allreduce_cnode, matmul_allreduce_cnode);
  MS_LOG(INFO) << "MatMulAllReduce replace success";
  return matmul_allreduce_cnode;
}
}  // namespace mindspore::opt
