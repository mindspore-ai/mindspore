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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_sigmoid_cast_add_fusion.h"
#include <vector>
#include <string>
#include <set>
#include "backend/common/pass/common/gllo_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "utils/ms_context.h"
#include "utils/phase.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
namespace {
const char elemwise_type[] = "sigmoid_add";
constexpr auto kFusedMatmulElemBinaryOpName = "FusedMatmulElemBinary";

bool IsAddNode(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsOneOfPrimitive(node, {prim::kPrimAdd, prim::kPrimBiasAdd})) {
      return true;
    }
  }
  return false;
}
}  // namespace

std::vector<std::string> MatMulSigmoidCastAddFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimMatMul->name(), prim::kPrimSigmoid->name(), prim::kPrimCast->name()};
  return ret;
}

const BaseRef MatMulSigmoidCastAddFusion::DefinePattern() const {
  auto x = std::make_shared<Var>();
  auto weight = std::make_shared<Var>();
  auto trans_a = std::make_shared<Var>();
  auto trans_b = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(x != nullptr, {});
  MS_CHECK_TRUE_RET(weight != nullptr, {});
  MS_CHECK_TRUE_RET(trans_a != nullptr, {});
  MS_CHECK_TRUE_RET(trans_b != nullptr, {});

  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto matmul_x_w = VectorRef({is_matmul, x, weight, trans_a, trans_b});

  auto is_sigmoid = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSigmoid>);
  VectorRef activation({is_sigmoid, matmul_x_w});
  auto is_cast0 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  VarPtr cast_input0 = std::make_shared<Var>();
  VectorRef cast0({is_cast0, activation, cast_input0});

  VarPtr is_add = std::make_shared<CondVar>(IsAddNode);
  VarPtr bias_input = std::make_shared<Var>();
  VectorRef bias_add({is_add, cast0, bias_input});

  auto is_cast1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimCast>);
  VarPtr cast_input1 = std::make_shared<Var>();
  VectorRef cast1({is_cast1, bias_add, cast_input1});
  return cast1;
}

const AnfNodePtr MatMulSigmoidCastAddFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }
  auto const &soc_ver = ms_context->ascend_soc_version();
  if (!soc_ver.empty() && soc_ver != "ascend910b" && soc_ver != "ascend910_93") {
    return nullptr;
  }

  auto net_phase = PhaseManager::GetInstance().phase();
  if (net_phase.rfind(kPhaseNamePrefill) == std::string::npos) {
    return nullptr;
  }

  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_matmul_sigmoid_cast_add =
    (std::find(enable_op_list.begin(), enable_op_list.end(), "MatMulSigmoidCastAdd") != enable_op_list.end());
  if (!enable_matmul_sigmoid_cast_add) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);

  auto cast_add_node = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast_add_node != nullptr, {});
  auto add_node = cast_add_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(add_node != nullptr, {});
  auto cast_sigmoid_node = add_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(cast_sigmoid_node != nullptr, {});
  auto sigmoid_node = cast_sigmoid_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(sigmoid_node != nullptr, {});
  MS_CHECK_TRUE_RET(sigmoid_node->func_graph() == add_node->func_graph(), {});
  auto matmul_cnode = sigmoid_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == add_node->func_graph(), {});
  // only support Add((?, ?), (?))
  if (common::AnfAlgo::GetPrevNodeOutputInferShape(add_node, 1).size() > 1) {
    return nullptr;
  }

  PrimitivePtr matmul_elem_prim = nullptr;
  matmul_elem_prim = std::make_shared<Primitive>(kFusedMatmulElemBinaryOpName);
  MS_CHECK_TRUE_RET(matmul_elem_prim, {});
  matmul_elem_prim->AddAttr("ElemwiseType", MakeValue(elemwise_type));

  auto input_trans_a = matmul_cnode->input(kIndex3)->cast<ValueNodePtr>();
  auto input_trans_b = matmul_cnode->input(kIndex4)->cast<ValueNodePtr>();
  matmul_elem_prim->AddAttr(kAttrIsTransA, input_trans_a->value());
  matmul_elem_prim->AddAttr(kAttrIsTransB, input_trans_b->value());

  auto input_x = matmul_cnode->input(kIndex1);
  auto input_weight = matmul_cnode->input(kIndex2);
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(input_x, support_dtype)) {
    return nullptr;
  }

  CNodePtr matmul_elemwise_cnode = nullptr;
  auto input_bias = add_node->input(kIndex2);
  matmul_elemwise_cnode = NewCNode({NewValueNode(matmul_elem_prim), input_x, input_weight, input_bias}, func_graph);
  MS_EXCEPTION_IF_NULL(matmul_elemwise_cnode);

  matmul_elemwise_cnode->set_scope(cast_add_node->scope());
  if (node->abstract() != nullptr) {
    matmul_elemwise_cnode->set_abstract(cast_add_node->abstract()->Clone());
  }

  return matmul_elemwise_cnode;
}
}  // namespace opt
}  // namespace mindspore
