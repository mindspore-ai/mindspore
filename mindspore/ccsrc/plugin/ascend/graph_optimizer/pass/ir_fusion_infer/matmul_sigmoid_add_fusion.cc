/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_sigmoid_add_fusion.h"
#include <vector>
#include <string>
#include <set>
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/phase.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kFusedMatmulElemBinaryOpName = "FusedMatmulElemBinary";
const char elemwise_type[] = "sigmoid_add";

bool IsAddNode(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsOneOfPrimitive(node, {prim::kPrimBiasAdd, prim::kPrimAdd})) {
      return true;
    }
  }
  return false;
}
}  // namespace

std::vector<std::string> MatMulSigmoidAddFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimMatMul->name()};
  return ret;
}

const BaseRef MatMulSigmoidAddFusion::DefinePattern() const {
  auto x = std::make_shared<Var>();
  auto w = std::make_shared<Var>();
  auto trans_a = std::make_shared<Var>();
  auto trans_b = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(x != nullptr, {});
  MS_CHECK_TRUE_RET(w != nullptr, {});
  MS_CHECK_TRUE_RET(trans_a != nullptr, {});
  MS_CHECK_TRUE_RET(trans_b != nullptr, {});

  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto matmul_x_w = VectorRef({is_matmul, x, w, trans_a, trans_b});

  auto is_sigmoid = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSigmoid>);
  VarPtr is_elem = std::make_shared<CondVar>(IsAddNode);
  VectorRef elem1({is_sigmoid, matmul_x_w});
  VarPtr elem_other_input2 = std::make_shared<Var>();
  VectorRef elem2({is_elem, elem1, elem_other_input2});
  return elem2;
}

const AnfNodePtr MatMulSigmoidAddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }
  auto const &soc_version = ms_context->ascend_soc_version();
  if (!soc_version.empty() && soc_version != "ascend910b" && soc_version != "ascend910_93") {
    return nullptr;
  }

  auto phase = PhaseManager::GetInstance().phase();
  if (phase.rfind(kPhaseNamePrefill) == std::string::npos) {
    return nullptr;
  }

  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_matmul_sigmoid_add =
    (std::find(enable_op_list.begin(), enable_op_list.end(), "MatMulSigmoidAdd") != enable_op_list.end());
  if (!enable_matmul_sigmoid_add) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);

  auto add_node = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_node != nullptr, {});
  auto sigmoid_node = add_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(sigmoid_node != nullptr, {});
  MS_CHECK_TRUE_RET(sigmoid_node->func_graph() == add_node->func_graph(), {});
  auto matmul_cnode = sigmoid_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == add_node->func_graph(), {});

  // only support Add((?, ?), (?))
  if (common::AnfAlgo::GetPrevNodeOutputInferShape(node, 1).size() > 1) {
    return nullptr;
  }
  // create op
  PrimitivePtr matmul_elemwise_prim = nullptr;
  matmul_elemwise_prim = std::make_shared<Primitive>(kFusedMatmulElemBinaryOpName);
  MS_CHECK_TRUE_RET(matmul_elemwise_prim, {});
  matmul_elemwise_prim->AddAttr("ElemwiseType", MakeValue(elemwise_type));

  auto input_trans_a = matmul_cnode->input(kIndex3)->cast<ValueNodePtr>();
  auto input_trans_b = matmul_cnode->input(kIndex4)->cast<ValueNodePtr>();
  matmul_elemwise_prim->AddAttr(kAttrIsTransA, input_trans_a->value());
  matmul_elemwise_prim->AddAttr(kAttrIsTransB, input_trans_b->value());

  auto input_x = matmul_cnode->input(kIndex1);
  auto input_w = matmul_cnode->input(kIndex2);
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(input_x, support_dtype)) {
    return nullptr;
  }

  CNodePtr matmul_elemwise_cnode = nullptr;
  auto input_e = add_node->input(kIndex2);
  matmul_elemwise_cnode = NewCNode({NewValueNode(matmul_elemwise_prim), input_x, input_w, input_e}, func_graph);
  MS_EXCEPTION_IF_NULL(matmul_elemwise_cnode);

  matmul_elemwise_cnode->set_scope(add_node->scope());
  if (node->abstract() != nullptr) {
    matmul_elemwise_cnode->set_abstract(add_node->abstract()->Clone());
  }

  return matmul_elemwise_cnode;
}
}  // namespace opt
}  // namespace mindspore
