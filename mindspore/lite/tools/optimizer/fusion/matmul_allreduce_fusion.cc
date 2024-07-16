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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/matmul_allreduce_fusion.h"
#include <set>
#include <memory>
#include <vector>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/all_reduce.h"
#include "mindspore/core/ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/ops_utils.h"
#include "mindspore/core/ir/anf.h"

namespace mindspore::opt {
std::unordered_map<std::string, VectorRef> MatMulAllReduceFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kPatternNameMatMulAllReduce] = DefineMatMulAllReducePattern();
  patterns[kPatternNameMatMulBiasAddAllReduce] = DefineMatMulBiasAddAllReducePattern();
  patterns[kPatternNameMatMulDequantAllReduce] = DefineMatMulDequantAllReducePattern();
  patterns[kPatternNameQuantBatchMatmulAllReduce] = DefineQuantBatchMatmulAllReducePattern();
  return patterns;
}

VectorRef MatMulAllReduceFusion::DefineMatMulAllReducePattern() const {
  auto matmul_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_1 != nullptr, {});
  auto matmul_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_2 != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  VectorRef matmul_ref = VectorRef({is_matmul, matmul_input_1, matmul_input_2});

  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_allreduce, matmul_ref});
  return pattern_ref;
}

VectorRef MatMulAllReduceFusion::DefineMatMulBiasAddAllReducePattern() const {
  auto matmul_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_1 != nullptr, {});
  auto matmul_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_2 != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMatMul>);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  VectorRef matmul_ref = VectorRef({is_matmul, matmul_input_1, matmul_input_2});

  auto biasadd_input = std::make_shared<Var>();
  auto is_biasadd = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimBiasAdd>);
  MS_CHECK_TRUE_RET(is_biasadd != nullptr, {});
  VectorRef biasadd_ref = VectorRef({is_biasadd, matmul_ref, biasadd_input});

  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_allreduce, biasadd_ref});
  return pattern_ref;
}

bool IsMatMulNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimMatMul) || CheckPrimitiveType(anf_node, prim::kPrimBatchMatMulV2) ||
           CheckPrimitiveType(anf_node, prim::kPrimMatMulV2);
  }
  return false;
}

VectorRef MatMulAllReduceFusion::DefineMatMulDequantAllReducePattern() const {
  auto matmul_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_1 != nullptr, {});
  auto matmul_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_2 != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsMatMulNode);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  VectorRef matmul_ref = VectorRef({is_matmul, matmul_input_1, matmul_input_2});

  auto dequant_input = std::make_shared<Var>();
  auto is_dequant = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimDequant>);
  MS_CHECK_TRUE_RET(is_dequant != nullptr, {});
  VectorRef dequant_ref = VectorRef({is_dequant, matmul_ref, dequant_input});

  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_allreduce, dequant_ref});
  return pattern_ref;
}

VectorRef MatMulAllReduceFusion::DefineQuantBatchMatmulAllReducePattern() const {
  MS_LOG(DEBUG) << "Do QuantBatchMatmulAllReduce Pattern";
  // QuantBatchMatmul
  auto x_input = std::make_shared<Var>();           // input x
  auto w_input = std::make_shared<Var>();           // input w
  auto deq_scale_input = std::make_shared<Var>();   // input dequant scale
  auto deq_offset_input = std::make_shared<Var>();  // input dequant offset
  auto deq_bias_input = std::make_shared<Var>();    // input dequant bias
  MS_CHECK_TRUE_RET(w_input != nullptr, {});
  MS_CHECK_TRUE_RET(x_input != nullptr, {});
  MS_CHECK_TRUE_RET(deq_scale_input != nullptr, {});
  MS_CHECK_TRUE_RET(deq_offset_input != nullptr, {});
  MS_CHECK_TRUE_RET(deq_bias_input != nullptr, {});
  auto is_quant_bmm = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantBatchMatmul>);
  MS_CHECK_TRUE_RET(is_quant_bmm != nullptr, {});

  auto quant_bmm = VectorRef({is_quant_bmm, x_input, w_input, deq_scale_input, deq_offset_input, deq_bias_input});

  // AllReduce
  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  auto allreduce = VectorRef({is_allreduce, quant_bmm});
  return allreduce;
}

PrimitivePtr MatMulAllReduceFusion::CreateMatMulAllReducePrim(const PrimitivePtr &allreduce_prim,
                                                              const PrimitivePtr &matmul_prim) const {
  auto matmul_allreduce_prim = prim::kPrimMatMulAllReduce->Clone();
  MS_CHECK_TRUE_RET(matmul_allreduce_prim, {});
  // add attr
  matmul_allreduce_prim->AddAttr(kAttrNameGroup, allreduce_prim->GetAttr(kAttrNameGroup));
  matmul_allreduce_prim->AddAttr(kAttrNameFusion, allreduce_prim->GetAttr(kAttrNameFusion));
  matmul_allreduce_prim->AddAttr(kAttrNameOp, allreduce_prim->GetAttr(kAttrNameOp));
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeA, matmul_prim->GetAttr(kAttrNameTransposeA));
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeB, matmul_prim->GetAttr(kAttrNameTransposeB));
  if (matmul_prim->HasAttr(kAttrNameNeedFusedXoffsetToBias)) {
    matmul_allreduce_prim->AddAttr(kAttrNameNeedFusedXoffsetToBias,
                                   matmul_prim->GetAttr(kAttrNameNeedFusedXoffsetToBias));
  }
  return matmul_allreduce_prim;
}

CNodePtr MatMulAllReduceFusion::CreateMatMulAllReduceNode(const FuncGraphPtr &func_graph,
                                                          const AnfNodePtr &node) const {
  MS_LOG(DEBUG) << "start create MatMulAllReduce";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto allreduce_cnode = node->cast<CNodePtr>();
  MS_ASSERT(allreduce_cnode != nullptr);
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  auto matmul_cnode = allreduce_cnode->input(kInputIndexOne)->cast<CNodePtr>();
  MS_ASSERT(matmul_cnode != nullptr);
  auto matmul_prim = GetCNodePrimitive(matmul_cnode);
  auto input_x_node = matmul_cnode->input(kInputIndexOne);
  MS_ASSERT(input_x_node != nullptr);
  auto input_y_node = matmul_cnode->input(kInputIndexTwo);
  MS_ASSERT(input_y_node != nullptr);

  if (IsMarkedTrainOp(allreduce_cnode) || IsMarkedTrainOp(matmul_cnode)) {
    return nullptr;
  }
  if (IsMultiOutputTensors(func_graph, matmul_cnode)) {
    return nullptr;
  }

  TypeId matmul_input_data_type;
  if (GetDataTypeFromAnfNode(input_x_node, &matmul_input_data_type) != RET_OK) {
    MS_LOG(ERROR) << "get input node data type failed." << input_x_node->fullname_with_scope();
    return nullptr;
  }
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  if (support_dtype.find(matmul_input_data_type) == support_dtype.end()) {
    MS_LOG(ERROR) << "unsupported data type: " << matmul_input_data_type;
    return nullptr;
  }

  auto matmul_allreduce_prim_c = CreateMatMulAllReducePrim(allreduce_prim, matmul_prim);
  std::vector<AnfNodePtr> matmul_allreduce_inputs = {input_x_node, input_y_node};

  auto matmul_allreduce_cnode = func_graph->NewCNode(matmul_allreduce_prim_c, matmul_allreduce_inputs);
  matmul_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  MS_LOG(DEBUG) << "create MatMulAllReduce success.";
  return matmul_allreduce_cnode;
}

CNodePtr MatMulAllReduceFusion::CreateMatMulBiasAddAllReduceNode(const FuncGraphPtr &func_graph,
                                                                 const AnfNodePtr &node) const {
  MS_LOG(DEBUG) << "start create MatMulAllBiasReduce";
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto allreduce_cnode = node->cast<CNodePtr>();
  MS_ASSERT(allreduce_cnode != nullptr);
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  auto biasAdd_cnode = allreduce_cnode->input(kInputIndexOne)->cast<CNodePtr>();
  MS_ASSERT(biasAdd_cnode != nullptr);
  auto matmul_cnode = biasAdd_cnode->input(kInputIndexOne)->cast<CNodePtr>();
  MS_ASSERT(matmul_cnode != nullptr);
  auto matmul_prim = GetCNodePrimitive(matmul_cnode);
  auto input_x_node = matmul_cnode->input(kInputIndexOne);
  MS_ASSERT(input_x_node != nullptr);
  auto input_y_node = matmul_cnode->input(kInputIndexTwo);
  MS_ASSERT(input_y_node != nullptr);
  auto input_bias_node = biasAdd_cnode->input(kInputIndexTwo);
  MS_ASSERT(input_bias_node != nullptr);

  if (IsMarkedTrainOp(allreduce_cnode) || IsMarkedTrainOp(matmul_cnode) || IsMarkedTrainOp(biasAdd_cnode)) {
    return nullptr;
  }
  if (IsMultiOutputTensors(func_graph, matmul_cnode) || IsMultiOutputTensors(func_graph, biasAdd_cnode)) {
    return nullptr;
  }

  TypeId matmul_input_data_type;
  if (GetDataTypeFromAnfNode(input_x_node, &matmul_input_data_type) != RET_OK) {
    MS_LOG(ERROR) << "get input node data type failed." << input_x_node->fullname_with_scope();
    return nullptr;
  }
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  if (support_dtype.find(matmul_input_data_type) == support_dtype.end()) {
    MS_LOG(ERROR) << "unsupported data type: " << matmul_input_data_type;
    return nullptr;
  }

  auto matmul_allreduce_prim_c = CreateMatMulAllReducePrim(allreduce_prim, matmul_prim);
  std::vector<AnfNodePtr> matmul_allreduce_inputs = {input_x_node, input_y_node, input_bias_node};

  auto matmul_allreduce_cnode = func_graph->NewCNode(matmul_allreduce_prim_c, matmul_allreduce_inputs);
  matmul_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  MS_LOG(DEBUG) << "create MatMulAllReduce success.";
  return matmul_allreduce_cnode;
}

CNodePtr MatMulAllReduceFusion::CreateMatMulDequantAllReduceNode(const FuncGraphPtr &func_graph,
                                                                 const AnfNodePtr &node) const {
  MS_LOG(INFO) << "start create MatMulDequantAllReduce";
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto allreduce_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(allreduce_cnode);
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  auto dequant_cnode = allreduce_cnode->input(kInputIndexOne)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dequant_cnode);
  auto matmul_cnode = dequant_cnode->input(kInputIndexOne)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(matmul_cnode);
  auto matmul_prim = GetCNodePrimitive(matmul_cnode);
  auto input_x_node = matmul_cnode->input(kInputIndexOne);
  MS_EXCEPTION_IF_NULL(input_x_node);
  auto input_y_node = matmul_cnode->input(kInputIndexTwo);
  MS_EXCEPTION_IF_NULL(input_y_node);
  AnfNodePtr input_bias_node;
  if (matmul_cnode->size() >= kInputIndexThree + 1) {
    input_bias_node = matmul_cnode->input(kInputIndexThree);
    MS_EXCEPTION_IF_NULL(input_bias_node);
  } else {
    auto none = std::make_shared<None>();
    input_bias_node = NewValueNode(none);
    MS_EXCEPTION_IF_NULL(input_bias_node);
    input_bias_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  }
  auto input_dequant_scale_node = dequant_cnode->input(kInputIndexTwo);
  MS_EXCEPTION_IF_NULL(input_dequant_scale_node);

  if (IsMarkedTrainOp(allreduce_cnode) || IsMarkedTrainOp(matmul_cnode) || IsMarkedTrainOp(dequant_cnode)) {
    return nullptr;
  }
  if (IsMultiOutputTensors(func_graph, matmul_cnode) || IsMultiOutputTensors(func_graph, dequant_cnode)) {
    return nullptr;
  }

  TypeId matmul_input_data_type;
  if (GetDataTypeFromAnfNode(input_x_node, &matmul_input_data_type) != RET_OK) {
    MS_LOG(ERROR) << "get input node data type failed." << input_x_node->fullname_with_scope();
    return nullptr;
  }
  const std::set<TypeId> support_dtype = {kNumberTypeInt8};
  if (support_dtype.find(matmul_input_data_type) == support_dtype.end()) {
    MS_LOG(ERROR) << "unsupported data type: " << matmul_input_data_type;
    return nullptr;
  }
  auto matmul_allreduce_prim_c = CreateMatMulAllReducePrim(allreduce_prim, matmul_prim);

  auto none = std::make_shared<None>();
  auto none_node = NewValueNode(none);
  MS_EXCEPTION_IF_NULL(none_node);
  none_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  func_graph->NewCNode({none_node});
  std::vector<AnfNodePtr> matmul_allreduce_inputs = {input_x_node, input_y_node, input_bias_node,         none_node,
                                                     none_node,    none_node,    input_dequant_scale_node};

  auto matmul_allreduce_cnode = func_graph->NewCNode(matmul_allreduce_prim_c, matmul_allreduce_inputs);
  matmul_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  MS_LOG(INFO) << "create MatMulDequantAllReduce success.";
  return matmul_allreduce_cnode;
}

CNodePtr MatMulAllReduceFusion::CreateQuantBatchMatmulAllReduceNode(const FuncGraphPtr &func_graph,
                                                                    const AnfNodePtr &node) const {
  MS_LOG(INFO) << "start create QuantBatchMatmulAllReduceNode";
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto allreduce_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(allreduce_cnode);

  auto qbmm_cnode = allreduce_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(qbmm_cnode);

  auto input_x = qbmm_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_x);
  auto input_w = qbmm_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_w);
  auto input_deq_scale = qbmm_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(input_deq_scale);
  auto input_deq_offset = qbmm_cnode->input(kIndex4);
  MS_EXCEPTION_IF_NULL(input_deq_offset);
  MS_LOG(WARNING) << "QuantBatchMatmul" << qbmm_cnode->fullname_with_scope()
                  << ", its dequant offset input will not use";
  auto input_deq_bias = qbmm_cnode->input(kIndex5);
  MS_EXCEPTION_IF_NULL(input_deq_bias);

  auto matmul_cnode_users = qbmm_cnode->func_graph()->manager()->node_users()[qbmm_cnode];
  MS_CHECK_TRUE_RET(matmul_cnode_users.size() == 1, {});

  // create op
  auto matmul_allreduce_prim = std::make_shared<Primitive>("MatMulAllReduce")->Clone();
  MS_CHECK_TRUE_RET(matmul_allreduce_prim, {});

  // add attr
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  auto qbmm_prim = GetCNodePrimitive(qbmm_cnode);
  matmul_allreduce_prim->AddAttr(kAttrNameGroup, allreduce_prim->GetAttr(kAttrNameGroup));
  matmul_allreduce_prim->AddAttr(kAttrNameFusion, allreduce_prim->GetAttr(kAttrNameFusion));
  matmul_allreduce_prim->AddAttr(kAttrNameOp, allreduce_prim->GetAttr(kAttrNameOp));
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeA, qbmm_prim->GetAttr(kAttrTransposeX1));
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeB, qbmm_prim->GetAttr(kAttrTransposeX2));

  auto none = std::make_shared<None>();
  auto none_node = NewValueNode(none);
  MS_EXCEPTION_IF_NULL(none_node);
  none_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  func_graph->NewCNode({none_node});
  auto matmul_allreduce_cnode =
    func_graph->NewCNode({NewValueNode(matmul_allreduce_prim), input_x, input_w, input_deq_bias, none_node, none_node,
                          none_node, input_deq_scale});
  if (matmul_allreduce_cnode == nullptr) {
    MS_LOG(ERROR) << "New matmul_allreduce_cnode should not be null, but it is null.";
    return nullptr;
  }
  matmul_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  matmul_allreduce_cnode->set_fullname_with_scope(allreduce_cnode->fullname_with_scope() + "_qbmm_allreduce_fusion");
  MS_LOG(INFO) << "create QuantBatchMatmulAllReduceNode success.";
  return matmul_allreduce_cnode;
}

AnfNodePtr MatMulAllReduceFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                          const mindspore::AnfNodePtr &node, const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto allreduce_cnode = node->cast<CNodePtr>();
  if (allreduce_cnode->size() != kInputSizeTwo) {
    return nullptr;
  }

  CNodePtr matmul_allreduce_cnode = nullptr;
  if (pattern_name == kPatternNameMatMulAllReduce) {
    matmul_allreduce_cnode = CreateMatMulAllReduceNode(func_graph, node);
  }
  if (pattern_name == kPatternNameMatMulBiasAddAllReduce) {
    matmul_allreduce_cnode = CreateMatMulBiasAddAllReduceNode(func_graph, node);
  }
  if (pattern_name == kPatternNameMatMulDequantAllReduce) {
    matmul_allreduce_cnode = CreateMatMulDequantAllReduceNode(func_graph, node);
  }
  if (pattern_name == kPatternNameQuantBatchMatmulAllReduce) {
    matmul_allreduce_cnode = CreateQuantBatchMatmulAllReduceNode(func_graph, node);
  }
  MS_CHECK_TRUE_RET(matmul_allreduce_cnode != nullptr, nullptr);

  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);

  // replace allreduce to MatMulAllReduce
  (void)manager->Replace(allreduce_cnode, matmul_allreduce_cnode);
  MS_LOG(DEBUG) << "MatMulAllReduce replace success";
  return matmul_allreduce_cnode;
}
}  // namespace mindspore::opt
