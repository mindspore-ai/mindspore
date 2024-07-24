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
#include "plugin/device/ascend/optimizer/ir_fusion/matmul_elemwise_fusion.h"
#include <vector>
#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
const BaseRef MatmulElemFusionBase::DefinePattern() const {
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

  VectorRef pattern = DefineMatmulFusionPattern(matmul_x_w);
  return pattern;
}

const AnfNodePtr MatmulElemFusionBase::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_matmul_elemwise =
    (std::find(enable_op_list.begin(), enable_op_list.end(), "MatMulElemwise") != enable_op_list.end());
  if (!enable_matmul_elemwise) {
    return nullptr;
  }
  if (elewise_input_num_ != kUnaryInputNum && elewise_input_num_ != kBinaryInputNum) {
    MS_LOG(EXCEPTION) << "Only support elewise unary and binary inputs";
  }

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);

  auto elemwise_node = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(elemwise_node != nullptr, {});
  auto matmul_cnode = elemwise_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == elemwise_node->func_graph(), {});

  // create op
  PrimitivePtr matmul_elemwise_prim = nullptr;
  if (elewise_input_num_ == kUnaryInputNum) {
    matmul_elemwise_prim = prim::kPrimFusedMatMulElemUnary->Clone();
  } else if (elewise_input_num_ == kBinaryInputNum) {
    matmul_elemwise_prim = prim::kPrimFusedMatMulElemBinary->Clone();
  }
  MS_CHECK_TRUE_RET(matmul_elemwise_prim, {});

  std::string elemwise_type = GetElemwiseType();
  if (elemwise_type == "bias_add" && common::AnfAlgo::GetOutputInferDataType(node, 0) != kFloat16->type_id()) {
    return nullptr;
  }
  matmul_elemwise_prim->AddAttr("ElemwiseType", MakeValue(elemwise_type));
  auto input_trans_a = matmul_cnode->input(kIndex3)->cast<ValueNodePtr>();
  auto input_trans_b = matmul_cnode->input(kIndex4)->cast<ValueNodePtr>();
  matmul_elemwise_prim->AddAttr(kAttrIsTransA, input_trans_a->value());
  matmul_elemwise_prim->AddAttr(kAttrIsTransB, input_trans_b->value());

  auto input_x = matmul_cnode->input(kIndex1);
  auto input_w = matmul_cnode->input(kIndex2);

  CNodePtr matmul_elemwise_cnode = nullptr;
  if (elewise_input_num_ == kUnaryInputNum) {
    matmul_elemwise_cnode = func_graph->NewCNode({NewValueNode(matmul_elemwise_prim), input_x, input_w});
  } else if (elewise_input_num_ == kBinaryInputNum) {
    auto input_e = elemwise_node->input(kIndex2);
    matmul_elemwise_cnode = func_graph->NewCNode({NewValueNode(matmul_elemwise_prim), input_x, input_w, input_e});
  }
  MS_EXCEPTION_IF_NULL(matmul_elemwise_cnode);

  matmul_elemwise_cnode->set_fullname_with_scope(elemwise_node->fullname_with_scope() + "_matmul_elemwise");
  if (node->abstract() != nullptr) {
    matmul_elemwise_cnode->set_abstract(elemwise_node->abstract()->Clone());
  }

  return matmul_elemwise_cnode;
}

const VectorRef MatmulElemBiasaddFusion::DefineMatmulFusionPattern(const VectorRef &predecessor) const {
  auto bias = std::make_shared<Var>();
  auto format = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(bias != nullptr, {});
  MS_CHECK_TRUE_RET(format != nullptr, {});

  auto is_biasadd = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBiasAdd>);
  MS_CHECK_TRUE_RET(is_biasadd != nullptr, {});
  auto biasadd_matmul_bias = VectorRef({is_biasadd, predecessor, bias, format});

  return biasadd_matmul_bias;
}

const VectorRef MatmulElemAddFusion::DefineMatmulFusionPattern(const VectorRef &predecessor) const {
  auto bias = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(bias != nullptr, {});

  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto matmul_add = VectorRef({is_add, predecessor, bias});

  return matmul_add;
}

const VectorRef MatmulElemReluFusion::DefineMatmulFusionPattern(const VectorRef &predecessor) const {
  auto is_relu = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimReLU>);
  MS_CHECK_TRUE_RET(is_relu != nullptr, {});

  auto relu_matmul = VectorRef({is_relu, predecessor});

  return relu_matmul;
}

const VectorRef MatmulElemGeluFusion::DefineMatmulFusionPattern(const VectorRef &predecessor) const {
  auto is_gelu = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimGeLU>);
  MS_CHECK_TRUE_RET(is_gelu != nullptr, {});

  auto gelu_matmul = VectorRef({is_gelu, predecessor});

  return gelu_matmul;
}
}  // namespace opt
}  // namespace mindspore
