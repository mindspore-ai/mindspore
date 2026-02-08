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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_elemwise_fusion.h"
#include <vector>
#include <string>
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/nn_ops.h"
#include "primitive/math_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kFusedMatmulElemUnaryOpName = "FusedMatmulElemUnary";
constexpr auto kFusedMatmulElemBinaryOpName = "FusedMatmulElemBinary";
constexpr size_t kUnaryInputNum = 1;
constexpr size_t kBinaryInputNum = 2;

bool IsElemNode(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsOneOfPrimitive(node,
                         {prim::kPrimBiasAdd, prim::kPrimAdd, prim::kPrimReLU, prim::kPrimGeLU, prim::kPrimFastGeLU})) {
      return true;
    }
  }

  return false;
}
}  // namespace

std::string MatmulElemFusion::GetElemwiseType(const CNodePtr &elemwise_node) const {
  static const std::map<std::string, std::string> kOpElemiseTypeMap = {{prim::kPrimBiasAdd->name(), "bias_add"},
                                                                       {prim::kPrimAdd->name(), "bias_add"},
                                                                       {prim::kPrimReLU->name(), "relu"},
                                                                       {prim::kPrimFastGeLU->name(), "fastgelu"},
                                                                       {prim::kPrimGeLU->name(), "gelu"}};
  return kOpElemiseTypeMap.at(common::AnfAlgo::GetCNodeName(elemwise_node));
}

std::vector<std::string> MatmulElemFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimMatMul->name()};
  return ret;
}

const BaseRef MatmulElemFusion::DefinePattern() const {
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

  VarPtr is_elem = std::make_shared<CondVar>(IsElemNode);
  VarPtr elem_other_input = std::make_shared<SeqVar>();
  VectorRef pattern({is_elem, matmul_x_w, elem_other_input});
  return pattern;
}

const AnfNodePtr MatmulElemFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto const &soc_version = ms_context->ascend_soc_version();
  const std::vector<std::string> valid_soc_version{"ascend910b", "ascend910_93", "ascend310p"};
  if (!soc_version.empty() &&
      (std::find(valid_soc_version.begin(), valid_soc_version.end(), soc_version) == valid_soc_version.end())) {
    return nullptr;
  }
  auto enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_matmul_elemwise =
    (std::find(enable_op_list.begin(), enable_op_list.end(), "MatMulElemwise") != enable_op_list.end());
  if (!enable_matmul_elemwise) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);

  auto elemwise_node = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(elemwise_node != nullptr, {});
  auto matmul_cnode = elemwise_node->input(kIndex1)->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(matmul_cnode != nullptr, {});
  MS_CHECK_TRUE_RET(matmul_cnode->func_graph() == elemwise_node->func_graph(), {});
  std::string elemwise_type = GetElemwiseType(elemwise_node);
  const std::string fastgelu_str = "fastgelu";
  if (elemwise_type != fastgelu_str && !soc_version.empty() && soc_version != "ascend910b" &&
      soc_version != "ascend910_93") {
    return nullptr;
  }
  const std::string bias_add_str = "bias_add";
  if (elemwise_type == bias_add_str && (common::AnfAlgo::GetPrevNodeOutputInferShape(node, 1).size() > 1 ||
                                        common::AnfAlgo::GetOutputInferDataType(node, 0) != kFloat16->type_id())) {
    return nullptr;
  }
  auto elewise_input_num = 0;
  if (elemwise_type == bias_add_str) {
    elewise_input_num = kBinaryInputNum;
  } else {
    elewise_input_num = kUnaryInputNum;
  }

  // create op
  PrimitivePtr matmul_elemwise_prim = nullptr;
  if (elewise_input_num == kUnaryInputNum) {
    matmul_elemwise_prim = std::make_shared<Primitive>(kFusedMatmulElemUnaryOpName);
  } else if (elewise_input_num == kBinaryInputNum) {
    matmul_elemwise_prim = std::make_shared<Primitive>(kFusedMatmulElemBinaryOpName);
  }
  MS_CHECK_TRUE_RET(matmul_elemwise_prim, {});
  matmul_elemwise_prim->AddAttr("ElemwiseType", MakeValue(elemwise_type));

  auto input_trans_a = matmul_cnode->input(kIndex3)->cast<ValueNodePtr>();
  auto input_trans_b = matmul_cnode->input(kIndex4)->cast<ValueNodePtr>();
  matmul_elemwise_prim->AddAttr(kAttrIsTransA, input_trans_a->value());
  matmul_elemwise_prim->AddAttr(kAttrIsTransB, input_trans_b->value());

  auto input_x = matmul_cnode->input(kIndex1);
  auto input_w = matmul_cnode->input(kIndex2);

  CNodePtr matmul_elemwise_cnode = nullptr;
  if (elewise_input_num == kUnaryInputNum) {
    matmul_elemwise_cnode = NewCNode({NewValueNode(matmul_elemwise_prim), input_x, input_w}, func_graph);
  } else if (elewise_input_num == kBinaryInputNum) {
    auto input_e = elemwise_node->input(kIndex2);
    matmul_elemwise_cnode = NewCNode({NewValueNode(matmul_elemwise_prim), input_x, input_w, input_e}, func_graph);
  }
  MS_EXCEPTION_IF_NULL(matmul_elemwise_cnode);

  matmul_elemwise_cnode->set_scope(elemwise_node->scope());
  if (node->abstract() != nullptr) {
    matmul_elemwise_cnode->set_abstract(elemwise_node->abstract()->Clone());
  }

  return matmul_elemwise_cnode;
}
}  // namespace opt
}  // namespace mindspore
