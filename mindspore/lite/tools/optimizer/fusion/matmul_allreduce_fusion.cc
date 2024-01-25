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
#include "ops/fusion/matmul_allreduce.h"
#include <set>
#include <memory>
#include <vector>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/all_reduce.h"
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

PrimitivePtr MatMulAllReduceFusion::CreateMatMulAllReducePrim(const PrimitivePtr &allreduce_prim,
                                                              const PrimitivePtr &matmul_prim) const {
  auto matmul_allreduce_prim = std::make_shared<ops::MatMulAllReduce>();
  MS_CHECK_TRUE_RET(matmul_allreduce_prim != nullptr, nullptr);
  // add attr
  auto matmul_allreduce_prim_c = matmul_allreduce_prim->GetPrim();
  MS_CHECK_TRUE_RET(matmul_allreduce_prim != nullptr, nullptr);
  matmul_allreduce_prim_c->AddAttr(kAttrNameCommRenuse, allreduce_prim->GetAttr(kAttrNameCommRenuse));
  matmul_allreduce_prim_c->AddAttr(kAttrNameGroup, allreduce_prim->GetAttr(kAttrNameGroup));
  matmul_allreduce_prim_c->AddAttr(kAttrNameFusion, allreduce_prim->GetAttr(kAttrNameFusion));
  matmul_allreduce_prim_c->AddAttr(kAttrNameOp, allreduce_prim->GetAttr(kAttrNameOp));
  matmul_allreduce_prim_c->AddAttr(kAttrNameTransposeA, matmul_prim->GetAttr(kAttrNameTransposeA));
  matmul_allreduce_prim_c->AddAttr(kAttrNameTransposeB, matmul_prim->GetAttr(kAttrNameTransposeB));
  return matmul_allreduce_prim_c;
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
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (support_dtype.find(matmul_input_data_type) == support_dtype.end()) {
    MS_LOG(ERROR) << "only support float16 and bfloat16 data type";
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
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (support_dtype.find(matmul_input_data_type) == support_dtype.end()) {
    MS_LOG(ERROR) << "only support float16 and bfloat16 data type";
    return nullptr;
  }

  auto matmul_allreduce_prim_c = CreateMatMulAllReducePrim(allreduce_prim, matmul_prim);
  std::vector<AnfNodePtr> matmul_allreduce_inputs = {input_x_node, input_y_node, input_bias_node};

  auto matmul_allreduce_cnode = func_graph->NewCNode(matmul_allreduce_prim_c, matmul_allreduce_inputs);
  matmul_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  MS_LOG(DEBUG) << "create MatMulAllReduce success.";
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
  MS_CHECK_TRUE_RET(matmul_allreduce_cnode != nullptr, nullptr);

  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);

  // replace allreduce to MatMulAllReduce
  (void)manager->Replace(allreduce_cnode, matmul_allreduce_cnode);
  MS_LOG(DEBUG) << "MatMulAllReduce replace success";
  return matmul_allreduce_cnode;
}
}  // namespace mindspore::opt
